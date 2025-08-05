# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from src.sharpdepth.data.datasets_and_samplers.base_depth_dataset import BaseDepthDataset, DepthFileNameMode, get_pred_name, DatasetMode


import numpy as np
import os
from PIL import Image

class UnrealStereoDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # KITTI data parameter
            min_depth=1e-5,
            max_depth=200,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )
        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]
        self.intrinsics = torch.tensor([ [1920.000000, 0, 1920.000000],
                                    [0, 1920.000000, 1080.0],
                                    [0,0,1]]).float()

    def _read_image(self, img_rel_path) -> np.ndarray:
        image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        
        image = Image.open(image_to_read)  # [H, W, rgb]
        image = np.asarray(image)[:, :, :3]
        return image


    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode KITTI depth
        depth_decoded = depth_in / 256.0
        return depth_decoded

    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        return rgb_data

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        outputs = {}
        disp = np.load(os.path.join(self.dataset_dir, depth_rel_path))
        ## Convert to Depth ##
        depth = 960.0/disp
        depth_raw_linear = torch.from_numpy(depth).float().unsqueeze(0).clamp(0, 200)  # [1, H, W]
        outputs["depth_raw_linear"] = depth_raw_linear.clone()
        outputs["depth_filled_linear"] = depth_raw_linear.clone()
        return outputs

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
