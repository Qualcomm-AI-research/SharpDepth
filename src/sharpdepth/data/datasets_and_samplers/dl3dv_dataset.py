# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from src.sharpdepth.data.datasets_and_samplers.base_depth_dataset import BaseDepthDataset, DepthFileNameMode, get_pred_name, DatasetMode
import numpy as np 
import os
from PIL import Image

class DL3DVDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            min_depth=1e-5,
            max_depth=100,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        return rgb_data


    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)

        rasters = {}
        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        filename_line = self.filenames[index]
        fx = float(filename_line[-4])
        fy = float(filename_line[-3])
        cx = float(filename_line[-2])
        cy = float(filename_line[-1])

        intrinsics = torch.tensor([[fx, 0, cx],
                                    [0, fy,cy],
                                    [0, 0,  1]]).float()

        other = {"index": index, "rgb_relative_path": rgb_rel_path, 'disp_name': self.disp_name, 'intrinsics': intrinsics, }

        return rasters, other


