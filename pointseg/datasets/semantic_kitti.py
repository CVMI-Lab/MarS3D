# * Imports and correspoding variables

# Common libs

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from pointseg.utils.logger import get_root_logger
from pointseg.datasets.builder import DATASETS
from pointseg.datasets.transform import Compose, TRANSFORMS
from scipy import sparse

# Dataset class

@DATASETS.register_module()
class SemanticKITTIDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data/semantic_kitti',
                 fusion_frame=[0,-1,-2],
                 learning_map=None,
                 transform=None,
                 cache_data=True,
                 test_mode=False,
                 test_cfg=None,
                 loop=1):
        super(SemanticKITTIDataset, self).__init__()
        
        self.data_root = data_root
        self.split = split
        self.split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        )
        
        self.add_frame = [0]
        if len(fusion_frame) > 1:
            self.add_frame = fusion_frame[1:]
        
        self.learning_map = learning_map[0] # Semantic-motion joint Learning map
        self.learning_map_b = learning_map[1] # Motion-aware Learning map
        self.learning_map_c = learning_map[2] # Semantic Learning map
        
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.cache_data = cache_data
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            pass

        if isinstance(self.split, str):
            seq_list = self.split2seq[split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += self.split2seq[split]
        else:
            raise NotImplementedError
        
        self.calibrations = {}
        self.poses = {}
        self.data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "sequences", seq)
            seq_files = sorted(
                os.listdir(os.path.join(seq_folder, "velodyne")))
            self.data_list += [os.path.join(seq_folder, "velodyne", file) for file in seq_files]
            self.calibrations[seq] = self.parse_calibration(os.path.join(seq_folder, "calib.txt"))
            poses_f64 = self.parse_poses(os.path.join(seq_folder, 'poses.txt'), self.calibrations[seq])
            self.poses[int(seq)] = [pose.astype(np.float32) for pose in poses_f64]
        
        self.data_idx = np.arange(len(self.data_list))
        logger = get_root_logger()
        logger.info("Totally {} x {} samples in {} set.".format(len(self.data_idx), self.loop, split))

    def parse_poses(self, filename, calibration):
        file = open(filename)
        poses = []
        tr = calibration["Tr"]
        tr_inv = np.linalg.inv(tr)
        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(np.matmul(tr_inv, np.matmul(pose, tr)))
        return poses

    def parse_calibration(self, filename):
        calibration = {}
        calibration_file = open(filename)
        for line in calibration_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calibration[key] = pose
        calibration_file.close()
        return calibration

    def calibrate_scan_pose(self, scan, pose, target_pose):
        points = np.hstack((scan[:, :3], np.ones_like(scan[:, :1])))
        remissions = scan[:, 3]
        diff = np.matmul(np.linalg.inv(target_pose), pose)
        points_target = np.matmul(diff, points.T).T
        points_target[:, 3] = remissions
        return points_target


    def prepare_train_data(self, idx):

        data_idx = self.data_idx[idx % len(self.data_idx)]
        with open(self.data_list[data_idx], 'rb') as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        scan_idx = int(os.path.basename(self.data_list[data_idx]).split(".")[0])
        seq_idx = int(os.path.basename(os.path.dirname(os.path.dirname(self.data_list[data_idx]))))
        label_file = self.data_list[data_idx].replace('velodyne', 'labels').replace('.bin', '.label')
        pose = self.poses[seq_idx][scan_idx]
        
        main_scan_num = scan.shape[0]
        main_scan_num = np.array(main_scan_num)

        # Pre-processing BEV parameters setting
        # %Currently, the per-pixel mapping size and the BEV mapping range is fixed.%
        # TODO: Config control for the primary parameters
        
        be_pixel_size = 0.2
        berange_x = [-50, 50]
        berange_y = [-30, 30]
        bemap_x = int((berange_x[1] - berange_x[0]) / be_pixel_size) + 1
        bemap_y = int((berange_y[1] - berange_y[0]) / be_pixel_size) + 1

        scan = np.insert(scan, scan.shape[1], values=0, axis=1)
        label_file = self.data_list[data_idx].replace('velodyne', 'labels').replace('.bin', '.label')
        
        if os.path.exists(label_file):
            with open(label_file, 'rb') as b1:
                labels_in = np.fromfile(b1, dtype=np.int32).reshape(-1)
        else:
            labels_in = np.zeros(scan.shape[0]).astype(np.int32)
        main_label = labels_in[:]

        for fuse_idx in self.add_frame:
            fuse_ebd_mark = int(-fuse_idx)
            if scan_idx + min(self.add_frame) >= 0:
                with open(self.data_list[data_idx +fuse_idx], 'rb') as b2:
                    pose_to_fuse = self.poses[seq_idx][scan_idx + fuse_idx]
                    file_to_fuse = b2
                    file_to_fuse_ = self.data_list[data_idx +fuse_idx]
                    label_file_to_fuse = file_to_fuse_.replace('velodyne', 'labels').replace(
                    '.bin', '.label')
                    scan_to_fuse_ = np.fromfile(file_to_fuse, dtype=np.float32).reshape(-1, 4)
            else:
                with open(self.data_list[data_idx], 'rb') as b2:
                    pose_to_fuse = self.poses[seq_idx][scan_idx - fuse_idx]
                    file_to_fuse = b2
                    file_to_fuse_ = self.data_list[data_idx]
                    label_file_to_fuse = file_to_fuse_.replace('velodyne', 'labels').replace(
                    '.bin', '.label')
                    scan_to_fuse_ = np.fromfile(file_to_fuse, dtype=np.float32).reshape(-1, 4)

            scan_to_fuse = self.calibrate_scan_pose(scan_to_fuse_, pose_to_fuse, pose)
            scan_to_fuse = np.insert(scan_to_fuse, scan_to_fuse.shape[1], values=fuse_ebd_mark, axis=1)

            if os.path.exists(label_file_to_fuse):
                with open(label_file_to_fuse, 'rb') as b3:
                    labels_to_fuse = np.fromfile(b3, dtype=np.int32).reshape(-1)
            else:
                labels_to_fuse = np.zeros(scan_to_fuse_.shape[0]).astype(np.int32)

            if len(scan_to_fuse) != 0:
                scan = np.concatenate((scan, scan_to_fuse), 0)
                labels_in = np.concatenate((labels_in, labels_to_fuse), 0)

        
        scan1 = np.zeros_like(scan)
        scan1[:, :3] = scan[:, :3]
        scan1[:, 3:] = scan[:, 3:]  # scan: 4-dim:xyza of each point

        scan1[:, 0][scan1[:, 0] < berange_x[0]] = berange_x[0]
        scan1[:, 0][scan1[:, 0] > berange_x[1]] = berange_x[1]
        scan1[:, 1][scan1[:, 1] < berange_y[0]] = berange_y[0]
        scan1[:, 1][scan1[:, 1] > berange_y[1]] = berange_y[1]
        pc_ = np.round(scan1[:, :3] / be_pixel_size).astype(np.int32)
        scan1[:, 0] = np.abs(scan1[:, 0] - (pc_[:, 0] * be_pixel_size)) * 2 / be_pixel_size  # + 1
        scan1[:, 1] = np.abs(scan1[:, 1] - (pc_[:, 1] * be_pixel_size)) * 2 / be_pixel_size  # + 1
        pc_ -= pc_.min(0, keepdims=1)  # make voxel location positive
        pc_tm_ = np.concatenate((pc_[:, :2], scan[:, 4:]), 1)


        scan_be = scan1[np.logical_and(pc_tm_[:, 0] < bemap_x, pc_tm_[:, 1] < bemap_y)]
        scan_be[scan_be > 1] = 1
        pc_be = pc_tm_[np.logical_and(pc_tm_[:, 0] < bemap_x, pc_tm_[:, 1] < bemap_y)]
        
        beam = scan[:, -2:-1]
        xy = pc_be[:, :2].astype(int).T.tolist()
 
        bemap_input = []
        for cur_tm in set(pc_tm_[:, 2]):
            for cur_axis in range(2):
                row = pc_be[pc_be[:, 2] == cur_tm][:, 0]
                col = pc_be[pc_be[:, 2] == cur_tm][:, 1]
                data_count = np.ones(row.shape[0])
                X_count = sparse.coo_matrix((data_count, (row, col)), shape=(bemap_x, bemap_y)).toarray()
                data = scan_be[pc_be[:, 2] == cur_tm][:, cur_axis]
                X = sparse.coo_matrix((data, (row, col)), shape=(bemap_x, bemap_y)).toarray()
                X[X_count == 0] = -1
                X_count[X_count == 0] = 1
                X_map = np.divide(X, X_count) 
                bemap_input.append(X_map)
            i = np.zeros((bemap_x,bemap_y))

            i[xy[0], xy[1]][:,np.newaxis] += beam 
            bemap_input.append(i)

        coord = scan[:, :3]
        strength = scan[:, -2:].reshape([-1, 2])
        
        label_file = self.data_list[data_idx].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                label = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            label = np.zeros(coord.shape[0]).astype(np.int32)
        label = np.vectorize(self.learning_map.__getitem__)(labels_in & 0xFFFF).astype(np.int64)
        main_label = np.vectorize(self.learning_map.__getitem__)(main_label & 0xFFFF).astype(np.int64)
        label_b = np.vectorize(self.learning_map_b.__getitem__)(labels_in & 0xFFFF).astype(np.int64)
        label_c = np.vectorize(self.learning_map_c.__getitem__)(labels_in & 0xFFFF).astype(np.int64)

        be_input = np.array(bemap_input)

        data_dict = dict(coord=coord, color=strength, label=label, label_b=label_b, label_c=label_c, be_input=be_input, main_num=main_scan_num, main_label=main_label)
        data_dict = self.transform(data_dict)
        return data_dict
    
    def prepare_test_data(self, idx):
        raise NotImplementedError

    def get_data_name(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_idx) * self.loop
