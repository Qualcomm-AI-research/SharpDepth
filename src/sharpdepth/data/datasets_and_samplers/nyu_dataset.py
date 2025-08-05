# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from:
# https://github.com/prs-eth/Marigold/blob/v0.1.4/src/dataset/nyu_dataset.py

import torch
import numpy as np
from src.sharpdepth.data.datasets_and_samplers.base_depth_dataset import BaseDepthDataset, DepthFileNameMode, get_pred_name, DatasetMode

class NYUDataset(BaseDepthDataset):
    def __init__(
        self,
        eigen_valid_mask: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            # NYUv2 dataset parameter
            min_depth=1e-3,
            max_depth=10.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_id,
            **kwargs,
        )

        self.eigen_valid_mask = eigen_valid_mask

        self.intrinsics = torch.tensor([[518.8579, 0, 325.58245],
                                        [0, 519.46961, 253.73617],
                                        [0, 0, 1]]).float()
    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        # Decode NYU depth
        depth_decoded = depth_in / 1000.0
        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = super()._get_valid_mask(depth)

        # Eigen crop for evaluation
        if self.eigen_valid_mask:
            eval_mask = torch.zeros_like(valid_mask.squeeze()).bool()
            eval_mask[45:471, 41:601] = 1
            eval_mask.reshape(valid_mask.shape)
            valid_mask = torch.logical_and(valid_mask, eval_mask)

        return valid_mask
    
    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp

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

            ## Get sparse depth ##
            sparse_depth = self.get_sparse_depth(rasters['depth_filled_linear'], 500)
            rasters["depth_filled_linear"] = sparse_depth

            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(
                rasters["depth_raw_linear"]
            ).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(
                rasters["depth_filled_linear"]
            ).clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path, 'disp_name': self.disp_name, 'intrinsics': self.intrinsics}

        return rasters, other

