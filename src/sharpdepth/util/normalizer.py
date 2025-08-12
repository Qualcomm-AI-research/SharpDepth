# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Code adapted from:
# https://github.com/prs-eth/Marigold/blob/main/src/util/depth_transform.py

import torch
import logging

class ScaleShiftNormalizer:
    """
    Use near and far plane to linearly normalize depth,
        i.e. d' = d * s + t,
        where near plane is mapped to `norm_min`, and far plane is mapped to `norm_max`
    Near and far planes are determined by taking quantile values.
    """

    is_absolute = False
    far_plane_at_max = True

    def __init__(
        self, norm_min=-1.0, norm_max=1.0, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.clip = clip

    def __call__(self, depth_linear, valid_mask=None):

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()

        # Take min and max
        max_values = torch.max(depth_linear.view(depth_linear.size(0), -1), dim=1)[0]
        min_values = torch.min(depth_linear.view(depth_linear.size(0), -1), dim=1)[0]

        max_values = max_values.view(depth_linear.size(0), 1, 1, 1)
        min_values = min_values.view(depth_linear.size(0), 1, 1, 1)

        # scale and shift
        depth_norm_linear = (depth_linear - min_values) / (max_values - min_values) * self.norm_range + self.norm_min

        if self.clip:
            depth_norm_linear = torch.clip(depth_norm_linear, self.norm_min, self.norm_max)
        
        return {"norm_depth": depth_norm_linear, \
                "min" : min_values, "max" : max_values}


    def scale_back(self, depth_norm):
        # scale to [0, 1]
        depth_linear = (depth_norm - self.norm_min) / self.norm_range
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        logging.warning(f"{self.__class__} is not revertible without GT")
        return self.scale_back(depth_norm=depth_norm)
