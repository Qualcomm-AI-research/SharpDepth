# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from:
# https://github.com/prs-eth/Marigold/blob/v0.1.4/src/dataset/__init__.py


import os
from .base_depth_dataset import *


from .kitti_dataset import KITTIDataset
from .nyu_dataset import NYUDataset
from .dl3dv_dataset import DL3DVDataset
from .waymo_dataset import WaymoDataset
from .sintel_dataset import SintelDataset
from .unrealstereo_dataset import UnrealStereoDataset
from .spring_dataset import SpringDataset

dataset_name_class_dict = {
    "kitti": KITTIDataset,
    "nyu_v2": NYUDataset,
    "dl3dv": DL3DVDataset,
    "waymo": WaymoDataset,
    "sintel": SintelDataset,
    "unrealstereo": UnrealStereoDataset,
    "spring": SpringDataset
}


def get_dataset(
    cfg_data_split, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseDepthDataset:
    if "mixed" == cfg_data_split.name:
        dataset_ls = [
            get_dataset(_cfg, base_data_dir, mode, **kwargs)
            for _cfg in cfg_data_split.dataset_list
        ]
        return dataset_ls
    elif cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=cfg_data_split.filenames,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset