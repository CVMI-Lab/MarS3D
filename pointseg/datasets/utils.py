import random
from collections.abc import Mapping, Sequence
import SharedArray as SA
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from pointseg.utils.logger import get_root_logger


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))

    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        if "coord" in batch[0]:
            for data in batch:
                data.update(offset=torch.tensor([data["coord"].shape[0]]))
            batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
            batch["offset"] = torch.cumsum(batch["offset"], dim=0)
        else:
            batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, max_batch_points=1e10, mix_prob=0):
    assert isinstance(batch[0], Mapping)  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    assert batch["offset"][0] <= max_batch_points  # at least the first scan can be added to batch
    for i in range(len(batch["offset"]) - 1):
        if batch["offset"][i+1] > max_batch_points:
            batch["offset"] = batch["offset"][:i+1]
            for key in batch.keys():
                if key != "offset":
                    # TODO: bug for data_metas
                    batch[key] = batch[key][:batch["offset"][-1]]
            break

    # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
    if random.random() < mix_prob:
        batch["offset"] = torch.cat([batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0)
    return batch


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c ** 2))
