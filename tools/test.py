import os
import time
import random
import numpy as np
import argparse
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from functools import partial
from tensorboardX import SummaryWriter

from pointseg.datasets import build_dataset
from pointseg.model import build_model
from pointseg.utils.misc import AverageMeter, intersection_and_union_gpu, find_free_port, make_dirs
from pointseg.datasets.utils import collate_fn, point_collate_fn
from pointseg.utils.optimizer import build_optimizer
from pointseg.utils.scheduler import build_scheduler
from pointseg.utils.losses import build_criteria
from pointseg.utils.config import Config, DictAction
from pointseg.utils.logger import get_root_logger


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('config',
                        type=str,
                        default='configs/s3dis/ptv2-base-1.py',
                        help='config file')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def main_process(cfg):
    return not cfg.multiprocessing_distributed or (
            cfg.multiprocessing_distributed and cfg.rank % cfg.num_gpus_per_node == 0)


def main():
    args = get_parser()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.train_gpu is None:
        cfg.train_gpu = [int(i) for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.train_gpu)

    if cfg.seed is None:
        cfg.seed = random.randint(0, 2 ** 16)

    set_seed(cfg.seed)

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed
    cfg.num_gpus_per_node = len(cfg.train_gpu)
    if len(cfg.train_gpu) == 1:
        cfg.sync_bn = False
        cfg.distributed = False
        cfg.multiprocessing_distributed = False

    make_dirs(cfg.save_path)
    cfg.dump(os.path.join(cfg.save_path, "config.py"))

    if cfg.cache_data:
        build_dataset(cfg.data.train)
        build_dataset(cfg.data.val)

    if cfg.multiprocessing_distributed:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        cfg.world_size = cfg.num_gpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=cfg.num_gpus_per_node, args=(cfg.num_gpus_per_node, cfg))
    else:
        main_worker(cfg.train_gpu, cfg.num_gpus_per_node, cfg)


def main_worker(gpu, num_gpus_per_node, cfg):
    best_metric = 0
    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * num_gpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)

    logger = get_root_logger(log_file=os.path.join(cfg.save_path, "train.log"), file_mode='a' if cfg.resume else 'w')
    logger.info(f"Config:\n{cfg.pretty_text}")

    if cfg.seed is not None:
        seed = cfg.workers * cfg.rank + cfg.seed
        set_seed(seed)

    # build model
    logger.info("=> Creating model ...")
    model = build_model(cfg.model)
    if cfg.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number params: {}".format(n_parameters))
    logger.info("Num classes: {}".format(cfg.data.num_classes))

    writer = None
    if main_process(cfg):
        writer = SummaryWriter(cfg.save_path)

    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / num_gpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / num_gpus_per_node)
        cfg.workers = int((cfg.workers + num_gpus_per_node - 1) / num_gpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters= False #True #cfg.find_unused_parameters
        )

    else:
        model = torch.nn.DataParallel(model.cuda())

    # build dataset & dataloader
    logger.info("=> Creating dataset & dataloader ...")
    train_data = build_dataset(cfg.data.train)

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=cfg.workers, rank=cfg.rank,
        seed=cfg.seed) if cfg.seed is not None else None

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=cfg.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg.workers,
                                               sampler=train_sampler,
                                               collate_fn=partial(point_collate_fn,
                                                                  max_batch_points=cfg.max_batch_points,
                                                                  mix_prob=cfg.mix_prob
                                                                  ),
                                               pin_memory=True,
                                               worker_init_fn=init_fn,
                                               drop_last=True,
                                               persistent_workers=True)

    val_loader = None
    if cfg.evaluate:
        val_data = build_dataset(cfg.data.val)
        if cfg.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=cfg.batch_size_val,
                                                 shuffle=False,
                                                 num_workers=cfg.workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler,
                                                 collate_fn=collate_fn)

    # Build criteria, optimize, scheduler
    logger.info("=> Creating criteria, optimize, scheduler, scaler(amp) ...")
    criteria = build_criteria(cfg.criteria)
    optimizer = build_optimizer(cfg.optimizer, model, cfg.param_dicts)
    cfg.scheduler.steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(cfg.scheduler, optimizer)
    logger.info("Update steps_per_epoch to {}".format(cfg.scheduler.steps_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if cfg.enable_amp else None

    logger.info("=> Checking weight & resume ...")
    resume_weight = ''
    logger.info("=> loading checkpoint '{}'".format(resume_weight))
    checkpoint = torch.load(resume_weight, map_location=lambda storage, loc: storage.cuda())
    cfg.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_weight, checkpoint['epoch']))

    for epoch in range(1):
        epoch_log = 0

        is_best = False
        
        if  cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = \
                validate(cfg, val_loader, model, criteria)
            current_metrics = dict(mIoU=mIoU_val, mAcc=mAcc_val, allAcc=allAcc_val)  # register metrics

    logger.info('==>Evaluation done!\nBest {}: {:.4f}'.format(cfg.metric, best_metric))
    if writer is not None:
        writer.close()


def validate(cfg, val_loader, model, criteria):
    logger = get_root_logger()
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    mh_weight = 1
    mh_dic = [19, -1, -1, 24, 23, 21, 20, 22]
    mh_dic_add = [0, 3, 4, 5, 6, 7]
    
    model.eval()
    end = time.time()
    for i, input_dict in enumerate(val_loader):
        data_time.update(time.time() - end)
        for key in input_dict.keys():
            if key != "data_metas":
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        coord = input_dict["coord"]
        feat = input_dict["feat"]
        target = input_dict["label"]
        offset = input_dict["offset"]
        mh = False
            
        with torch.no_grad():
            outputs = model(input_dict)


        if isinstance(outputs, tuple):
            mh = True
            outputs_c, outputs_b = outputs
        else:
            outputs_c = outputs
        if mh:
            targets_c = input_dict['label_c']
            targets_b = input_dict['label_b']
            if outputs_c.requires_grad:
                loss = criteria(outputs_c, targets_c) + mh_weight * criteria(outputs_b, targets_b)
            else:
                loss = criteria(outputs_c, targets_c)
        else:
            loss = criteria(outputs_c, target)

        if mh:
            output = outputs_c.max(1)[1]
            output_b = outputs_b.max(1)[1]
            for i_cat in range(6):
                output[np.logical_and((output == mh_dic_add[i_cat]).cpu().numpy(), (output_b == 1).cpu().numpy())] = mh_dic[mh_dic_add[i_cat]]
        else:
            output = outputs.max(1)[1]
            
        n = coord.size(0)
        if cfg.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        # For fairness
        cur_length_point = input_dict['length'].data.cpu().numpy()
        cur_length_voxel = input_dict['count'].data.cpu().numpy()
        output1 = output[:cur_length_voxel[0]]
        cur_inverse_1 = input_dict['inverse'].data.cpu().numpy()[:cur_length_point[0]]
        main_points_1 = input_dict['main_num'].data.cpu().numpy()[0]
        output2 = output1[cur_inverse_1][:main_points_1]
        target2 = input_dict['main_label']

        intersection, union, target = \
            intersection_and_union_gpu(output2, target2, cfg.data.num_classes, cfg.data.ignore_label)
        if cfg.multiprocessing_distributed:
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % cfg.log_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(cfg.data.num_classes):
        logger.info('Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
            idx=i, name=cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc

    gc.collect()
    main()
