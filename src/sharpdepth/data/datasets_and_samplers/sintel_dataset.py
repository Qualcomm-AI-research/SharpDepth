# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from src.sharpdepth.data.datasets_and_samplers.base_depth_dataset import BaseDepthDataset, DepthFileNameMode, get_pred_name, DatasetMode
import numpy as np
import os
import glob 
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth


class SintelDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            min_depth=1e-3,
            max_depth=200.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_id,
            **kwargs,
        )

        self.intrinsics = torch.tensor([[518.8579, 0, 325.58245],
                                        [0, 519.46961, 253.73617],
                                        [0, 0, 1]]).float()
        
         
        self.scenes = sorted(os.listdir(os.path.join(self.dataset_dir, "training", "final")))
        self.img_path = []
        self.depth_path = []
        for s in self.scenes:
            files = os.listdir(os.path.join(self.dataset_dir, "training", "final", s))
            for f in files:
                self.img_path.append(os.path.join("training", "final", s, f))
                self.depth_path.append(os.path.join(self.dataset_dir, "training", "depth", s, f.replace('png', 'dpt')))

    def _read_depth_file(self, rel_path):
        depth_in = depth_read(rel_path)
        return depth_in

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = super()._get_valid_mask(depth)
        return valid_mask

    def _get_data_path(self, index):
        # Get data path
        rgb_rel_path = self.img_path[index]

        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = self.depth_path[index]
            filled_rel_path = None
            
        return rgb_rel_path, depth_rel_path, filled_rel_path

    def __len__(self):
        return len(self.img_path)

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)
        
        rasters = {}
        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path
            )
            rasters.update(depth_data)
            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]
            ).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]
            ).clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path, 'disp_name': self.disp_name, 'intrinsics': self.intrinsics}

        return rasters, other
