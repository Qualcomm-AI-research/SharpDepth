# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Code adapted from:
# https://github.com/prs-eth/Marigold/blob/v0.1.4/infer.py

import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.sharpdepth.data.datasets_and_samplers.base_depth_dataset import BaseDepthDataset, DepthFileNameMode, get_pred_name, DatasetMode
from src.sharpdepth.data.datasets_and_samplers import get_dataset
from src.sharpdepth.pipeline.lotus_pipeline import LotusGPipeline
from diffusers import UNet2DConditionModel
from unidepth.models import UniDepthV1
from src.sharpdepth.util.alignment import align_depth_least_square, depth2disparity, disparity2depth

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    half_precision = args.half_precision

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"dataset config = `{dataset_config}`."
    )

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipeline = LotusGPipeline.from_pretrained("jingheya/lotus-depth-g-v1-0")
    unidepth = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

    pipeline = pipeline.to(device)
    unidepth = unidepth.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True
        ):
            # Read input image
            test_image = batch['rgb_int']
            predictions = unidepth.infer(test_image.squeeze().int())

            test_image = test_image / 127.5 - 1.0 
            test_image = test_image.to(device)

            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

            # Run
            pred = pipeline(
                rgb_in=test_image, 
                prompt='', 
                num_inference_steps=1, 
                # guidance_scale=0,
                output_type='np',
                timesteps=[999],
                task_emb=task_emb,
                ).images[0]

            output_npy = pred.mean(axis=-1)
            unidepth_pred = predictions['depth'].squeeze().cpu().numpy()
            final_pred, scale, shift = align_depth_least_square(
                                                            gt_arr=unidepth_pred,
                                                            pred_arr=output_npy,
                                                            valid_mask_arr=np.ones(unidepth_pred.shape, dtype=np.bool_),
                                                            return_scale_shift=True,
                                                            max_resolution=None,
                                                    )

            # Save predictions
            rgb_filename = batch["rgb_relative_path"][0]
            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(output_dir, os.path.dirname(rgb_filename))
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            pred_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=".npy")
            save_to = os.path.join(scene_dir, pred_basename)
            
            np.save(save_to, final_pred)
