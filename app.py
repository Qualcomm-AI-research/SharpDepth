# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from unidepth.models import UniDepthV1

from src.sharpdepth.data.datasets_and_samplers import get_dataset
from src.sharpdepth.data.datasets_and_samplers.base_depth_dataset import (
    BaseDepthDataset,
    DatasetMode,
    DepthFileNameMode,
    get_pred_name,
)
from src.sharpdepth.pipeline.pipeline import SharpDepthPipeline

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using SharpDepth."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")

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
    output_dir = args.output_dir
    input_dir = args.input_dir
    half_precision = args.half_precision

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(f"Inference settings: checkpoint = `{checkpoint_path}`, ")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(f"Running with half precision ({dtype}), might lead to suboptimal result.")
    else:
        dtype = torch.float32
        variant = None

    pipeline = SharpDepthPipeline.from_pretrained(checkpoint_path)
    unidepth = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

    pipeline = pipeline.to(device, dtype=dtype)
    unidepth = unidepth.to(device, dtype=dtype)

    imgs = sorted(os.listdir(input_dir))
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(imgs):
            # Read input image
            rgb = Image.open(os.path.join(input_dir, batch))
            out = pipeline(rgb, unidepth, processing_res=768, denoising_steps=1)
            depth_colored = out.depth_colored
            unidepth_colored = out.unidepth_colored

            depth_colored.save(os.path.join(output_dir, batch))
            unidepth_colored.save(os.path.join(output_dir, batch.split(".")[0] + "_uni.jpg"))
