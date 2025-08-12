# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Code adapted from:
# https://github.com/prs-eth/Marigold/blob/v0.1.4/src/util/metric.py

import pandas as pd
import torch
import numpy as np
from scipy import ndimage
from skimage import feature

# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()



def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss



### Sharpness evaluation ###
def depth2disparity(depth, return_mask=False):
    inverse_depth = 1.0 / depth.clip(min=1e-6)
    return inverse_depth

def dbe_acc(output, target, valid_mask=None):
    depth = output.cpu().numpy()
    gt = target.cpu().numpy()


    ## Run for Depth ##
    pred_normalized = gt - np.nanmin(gt)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    gt_depth_edge = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    pred_normalized = depth - np.nanmin(depth)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    pred_depth_edge = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    ## Run for Disparity ##
    disp_gt = depth2disparity(gt)
    pred_normalized = disp_gt - np.nanmin(disp_gt)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    gt_disp_edge = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    disp = depth2disparity(depth)
    pred_normalized = disp - np.nanmin(disp)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    pred_disp_edge = feature.canny(pred_normalized,sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    ## MERGE ##
    gt_edge = (gt_depth_edge | gt_disp_edge)
    pred_edge = (pred_depth_edge | pred_disp_edge)

    # compute distance transform for chamfer metric
    D_gt = ndimage.distance_transform_edt(1-gt_edge)
    D_est = ndimage.distance_transform_edt(1-pred_edge)

    max_dist_thr = 10.; # Threshold for local neighborhood

    mask_D_gt = D_gt<max_dist_thr; # truncate distance transform map

    E_fin_est_filt = pred_edge*mask_D_gt; # compute shortest distance for all predicted edges

    if np.sum(E_fin_est_filt) == 0: # assign MAX value if no edges could be found in prediction
        dbe_acc = max_dist_thr
    else:
        dbe_acc = np.nansum(D_gt*E_fin_est_filt)/np.nansum(E_fin_est_filt) # accuracy: directed chamfer distance

    return torch.tensor(dbe_acc)


def dbe_compl(output, target, valid_mask=None):
    depth = output.cpu().numpy()
    gt = target.cpu().numpy()

    ## Run for Depth ##
    pred_normalized = gt - np.nanmin(gt)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    gt_depth_edge = feature.canny(pred_normalized,
                            sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    pred_normalized = depth - np.nanmin(depth)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    pred_depth_edge = feature.canny(pred_normalized,
                            sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    ## Run for Disparity ##
    disp_gt = depth2disparity(gt)
    pred_normalized = disp_gt - np.nanmin(disp_gt)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    gt_disp_edge = feature.canny(pred_normalized,
                            sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)

    disp = depth2disparity(depth)
    pred_normalized = disp - np.nanmin(disp)
    pred_normalized = pred_normalized/np.nanmax(pred_normalized)
    pred_disp_edge = feature.canny(pred_normalized,
                            sigma=np.sqrt(2),low_threshold=0.1,high_threshold=0.2)



    ## MERGE ##
    gt_edge = (gt_depth_edge | gt_disp_edge)
    pred_edge = (pred_depth_edge | pred_disp_edge)
    
    # compute distance transform for chamfer metric
    D_gt = ndimage.distance_transform_edt(1-gt_edge)
    D_est = ndimage.distance_transform_edt(1-pred_edge)

    max_dist_thr = 10.; # Threshold for local neighborhood

    mask_D_gt = D_gt<max_dist_thr; # truncate distance transform map

    E_fin_est_filt = pred_edge*mask_D_gt; # compute shortest distance for all predicted edges

    if np.sum(E_fin_est_filt) == 0: # assign MAX value if no edges could be found in prediction
        dbe_com = max_dist_thr
    else:
        dbe_com = np.nansum(D_est*gt_edge)/np.nansum(gt_edge) # completeness: directed chamfer distance (reversed)

    return torch.tensor(dbe_com)
