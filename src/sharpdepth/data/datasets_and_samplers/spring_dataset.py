# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from src.sharpdepth.data.datasets_and_samplers.base_depth_dataset import BaseDepthDataset, DepthFileNameMode, get_pred_name, DatasetMode

import numpy as np
import os

SPRING_BASELINE = 0.065

from src.sharpdepth.data.datasets_and_samplers.flow_io import *

def get_depth(disp1, intrinsics, baseline=SPRING_BASELINE):
    """
    get depth from reference frame disparity and camera intrinsics
    """
    return intrinsics * baseline / disp1


class SpringDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            min_depth=1e-5,
            max_depth=200,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )
        # Filter out empty depth
        self.filenames = [f for f in self.filenames if "None" != f[1]]
        self.dataset_dir_depth = "../unidepth_exp/eval/unidepthv1_with_intrinsics/spring/prediction"
    
    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs


    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = filename_line[1]
            try:
                filled_rel_path = filename_line[2]
            except:
                filled_rel_path = None

        return rgb_rel_path, depth_rel_path, float(filename_line[2]), float(filename_line[3]), float(filename_line[4]), float(filename_line[5])


    def _load_rgb_data(self, rgb_rel_path):
        rgb_data = super()._load_rgb_data(rgb_rel_path)
        return rgb_data

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        outputs = {}
        disp = readDispFile(os.path.join(self.dataset_dir, depth_rel_path))
        disp = disp[::2,::2]
        depth = get_depth(disp, float(filled_rel_path))
        depth_raw_linear = torch.from_numpy(depth).float().unsqueeze(0).clamp(0, 200)  # [1, H, W]
        outputs["depth_raw_linear"] = depth_raw_linear.clone()
        outputs["depth_filled_linear"] = depth_raw_linear.clone()
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, fx, fy, cx, cy = self._get_data_path(index=index)

        rasters = {}
        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(
                depth_rel_path=depth_rel_path, filled_rel_path=fx
            )
            rasters.update(depth_data)

            try:
                rgb_basename = os.path.basename(rgb_rel_path)
                scene_dir = os.path.join(self.dataset_dir_depth, os.path.dirname(rgb_rel_path))
                pred_basename = get_pred_name(rgb_basename, self.name_mode, suffix=".npy")
                unidepth_path = os.path.join(scene_dir, pred_basename)
                rasters["depth_filled_linear"] = torch.from_numpy(np.load(unidepth_path)).unsqueeze(0)            
            except:
                pass
                
            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]
            ).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]
            ).clone()
        
        intrinsics = torch.tensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]]).float()
        other = {"index": index, "rgb_relative_path": rgb_rel_path, 'disp_name': self.disp_name, 'intrinsics': intrinsics}

        return rasters, other
    

