# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import itertools
import logging
import math
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import diffusers
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from ema_pytorch import EMA
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from unidepth.models import UniDepthV1

from src.sharpdepth.data.datasets_and_samplers import BaseDepthDataset, DatasetMode, get_dataset
from src.sharpdepth.data.datasets_and_samplers.mixed_sampler import MixedBatchSampler
from src.sharpdepth.pipeline.pipeline import SharpDepthPipeline
from src.sharpdepth.util.config_util import find_value_in_omegaconf, recursive_load_config
from src.sharpdepth.util.logging_util import config_logging
from src.sharpdepth.util.normalizer import ScaleShiftNormalizer

logger = get_logger(__name__)


@torch.no_grad()
def encode_image(vae, rgb):
    rgb_latent = vae.encode(rgb).latent_dist.mean
    rgb_latent = rgb_latent * vae.config.scaling_factor
    return rgb_latent


@torch.no_grad()
def decode_image(vae, latent):
    # scale latent
    latent = latent / vae.config.scaling_factor
    # decode
    z = vae.post_quant_conv(latent)
    rgb = vae.decoder(z)
    return rgb


@torch.no_grad()
def encode_depth(vae, depth):
    depth_latent = vae.encode(depth.repeat(1, 3, 1, 1)).latent_dist.mean
    depth_latent = depth_latent * vae.config.scaling_factor
    return depth_latent


def l1_loss(pred_depth, gt_depth, mask):
    l1_loss = torch.abs(pred_depth - gt_depth) * mask
    l1_loss = l1_loss.sum() / (mask.sum() + 1e-8)
    return l1_loss


def abs_relative_difference_full(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    return abs_relative_diff


def encode_empty_text(tokenizer, text_encoder):
    """
    Encode text embedding for empty prompt
    """
    prompt = ""
    text_inputs = tokenizer(
        prompt,
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    empty_text_embed = text_encoder(text_input_ids)[0]

    return empty_text_embed


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def colorize(value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"):
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img


if "__main__" == __name__:
    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train your little sharpener!")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to save checkpoints"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--base_data_dir", type=str, default=None, help="directory of training data"
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="directory of pretrained checkpoint",
    )
    parser.add_argument(
        "--student_ckpt_dir",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="directory of pretrained checkpoint",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--depth_weight",
        type=float,
        default=0.2,
        help="Depth loss weight.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="marigold_train_t2i_adapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    args = parser.parse_args()
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir if args.base_data_dir is not None else os.environ["BASE_DATA_DIR"]
    )
    base_ckpt_dir = (
        args.base_ckpt_dir if args.base_ckpt_dir is not None else os.environ["BASE_CKPT_DIR"]
    )
    student_ckpt_dir = args.student_ckpt_dir

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # ---------------------------------------------------------------------

    # -------------------- Initialization --------------------
    cfg = recursive_load_config(args.config)
    # Full job name
    pure_job_name = os.path.basename(args.config).split(".")[0]
    # Add time prefix
    if args.add_datetime_prefix:
        job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
    else:
        job_name = pure_job_name
    # ---------------------------------------------------------------------

    # -------------------- Initialize Logger and Accelerator --------------------
    logging_dir = Path(output_dir, job_name)
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # ---------------------------------------------------------------------

    # -------------------- set seed --------------------
    if args.seed is not None:
        set_seed(args.seed)
    # ---------------------------------------------------------------------

    # -------------------- create logging folder --------------------
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
    # ---------------------------------------------------------------------

    # -------------------- unwrap function --------------------
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ---------------------------------------------------------------------

    # -------------------- create custom loading hooks --------------------
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema:
                data = {"ema": ema_model.state_dict()}
                torch.save(data, os.path.join(output_dir, "unet_ema.pt"))
                del data

            for model in models:
                sub_dir = (
                    "unet"
                    if isinstance(model, type(unwrap_model(student_unet)))
                    else "text_encoder"
                )
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            data = torch.load(os.path.join(input_dir, "unet_ema.pt"), map_location="cpu")
            ema_model.load_state_dict(data["ema"], strict=False)
            del data

        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    # ---------------------------------------------------------------------

    # -------------------- Logging settings --------------------
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        config_logging(cfg.logging, out_dir=logging_dir)
        logger.info(f"config: {cfg}")
    # ---------------------------------------------------------------------

    # -------------------- Set training device --------------------
    device = accelerator.device
    logger.info(f"device = {device}")
    # ---------------------------------------------------------------------

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = args.train_batch_size * args.gradient_accumulation_steps
    logger.info(
        f"Effective batch size: {eff_bs}, accumulation steps: {args.gradient_accumulation_steps}"
    )
    # ---------------------------------------------------------------------

    # -------------------- Create dataset --------------------
    if args.seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(args.seed)

    cfg_data = cfg.dataset

    # Training dataset
    train_dataset: BaseDepthDataset = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
    )

    if "mixed" == cfg_data.train.name:
        for data in train_dataset:
            if len(data) == 0:
                breakpoint()
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=args.train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_dataloader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_train_workers,
        )

    # Validation dataset
    val_loaders: List[DataLoader] = []
    for _val_dic in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_val_workers,
        )
        val_loaders.append(_val_loader)
    logger.info("Finish loading dataset")
    # ---------------------------------------------------------------------

    # -------------------- Model --------------------
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, subfolder="tokenizer")
    text_encoder_cls = import_model_class_from_model_name_or_path(base_ckpt_dir, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(
        base_ckpt_dir, subfolder="text_encoder", revision=None
    )
    noise_scheduler = DDPMScheduler.from_pretrained(base_ckpt_dir, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(base_ckpt_dir, subfolder="vae", revision=None)
    lotus_unet = UNet2DConditionModel.from_pretrained(
        base_ckpt_dir, subfolder="unet", revision=None
    )
    unidepth = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

    vae.requires_grad_(False)
    lotus_unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    student_unet = UNet2DConditionModel.from_pretrained(
        student_ckpt_dir, subfolder="unet", revision=None
    )
    student_unet.requires_grad_(True)
    # ---------------------------------------------------------------------

    # -------------------- EMA model --------------------
    if args.use_ema:
        ema_model = EMA(
            student_unet,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10,  # how often to actually update, to save on compute (updates every 10th .update() call)
        )
    # ---------------------------------------------------------------------

    # -------------------- XFORMER --------------------
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            logger.info("enable xformers memory efficient attention")
            lotus_unet.enable_xformers_memory_efficient_attention()
            student_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    # ---------------------------------------------------------------------

    # -------------------- Gradient checkpointing --------------------
    if args.gradient_checkpointing:
        logger.info("Gradient checkpointing")
        student_unet.enable_gradient_checkpointing()  # only student unet require grad
    # ---------------------------------------------------------------------

    # -------------------- Sanity check --------------------
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unwrap_model(student_unet).dtype != torch.float32:
        raise ValueError(
            f"Student unet loaded as datatype {unwrap_model(student_unet).dtype}. {low_precision_error_string}"
        )
    if unwrap_model(lotus_unet).dtype != torch.float32:
        raise ValueError(
            f"Lotus unet loaded as datatype {unwrap_model(lotus_unet).dtype}. {low_precision_error_string}"
        )
    # ---------------------------------------------------------------------

    # -------------------- Scale LR --------------------
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
    # ---------------------------------------------------------------------

    # -------------------- Set up optimizer --------------------
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, student_unet.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # ---------------------------------------------------------------------

    # -------------------- Set up training step and LR scheduler --------------------
    # Scheduler and math around the numfprobfber of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # ---------------------------------------------------------------------

    # -------------------- Prepare and move to cuda --------------------
    student_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student_unet, optimizer, train_dataloader, lr_scheduler
    )
    # ---------------------------------------------------------------------

    # -------------------- Set up training precision --------------------
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # ---------------------------------------------------------------------

    # -------------------- Move freezed model to GPU --------------------
    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=weight_dtype)
    lotus_unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    student_unet.to(dtype=weight_dtype)
    unidepth.to(accelerator.device)

    if args.use_ema:
        ema_model = ema_model.to(accelerator.device, dtype=weight_dtype)
    # ---------------------------------------------------------------------

    # -------------------- Precompute null & task embedding --------------------
    empty_text_emb = encode_empty_text(tokenizer, text_encoder).to(
        accelerator.device, dtype=weight_dtype
    )
    del tokenizer
    del text_encoder
    torch.cuda.empty_cache()
    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    # -------------------- Recalculate training step --------------------
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # ---------------------------------------------------------------------

    # The trackers initializes automatically on the main process.
    # -------------------- Initializes tracker --------------------
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    # ---------------------------------------------------------------------

    # -------------------- Trainer --------------------
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {sum([len(dataset) for dataset in train_dataset])}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # ---------------------------------------------------------------------

    # -------------------- Resume --------------------
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    # ---------------------------------------------------------------------

    # -------------------- TQDM & stuff --------------------
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    device = accelerator.device

    alphas_cumprod = noise_scheduler.alphas_cumprod
    alphas_cumprod = alphas_cumprod.to(device)
    depth_normalizer = ScaleShiftNormalizer()
    # ---------------------------------------------------------------------

    for epoch in range(first_epoch, args.num_train_epochs):
        student_unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(student_unet):
                rgb = batch["rgb_int"].to(weight_dtype) / 255.0

                ## UniDepth ##
                intrinsics = batch["intrinsics"].squeeze(0)

                with torch.no_grad():
                    rgb_unidepth = (rgb.squeeze(0) * 255.0).long()
                    predictions = unidepth.infer(rgb_unidepth)

                    disp_unidepth = predictions["depth"]
                    # disp_unidepth = 1/disp_unidepth.clamp(min=1e-6)

                normalize_obj = depth_normalizer(disp_unidepth)
                norm_disp_unidepth = normalize_obj["norm_depth"].to(dtype=weight_dtype)

                unidepth_latent = encode_depth(vae, norm_disp_unidepth)

                rgb = rgb * 2 - 1

                ## Lotus ##
                # Encode image
                with torch.no_grad():
                    rgb_latent = encode_image(vae, rgb)
                    lotus_timesteps = torch.ones((rgb_latent.shape[0],), device=device) * (
                        noise_scheduler.config.num_train_timesteps - 1
                    )
                    lotus_timesteps = lotus_timesteps.long()
                    batch_empty_text_embed = empty_text_emb.repeat((rgb_latent.shape[0], 1, 1)).to(
                        device, dtype=weight_dtype
                    )
                    batch_task_emb = task_emb.repeat((rgb_latent.shape[0], 1)).to(
                        device, dtype=weight_dtype
                    )

                # ---------------------------------
                # extract mask
                with torch.no_grad():
                    lotus_input = torch.cat(
                        [rgb_latent.detach(), torch.randn_like(rgb_latent)], dim=1
                    )  # this order is important

                    if args.use_ema:
                        lotus_pred = ema_model(
                            lotus_input,
                            lotus_timesteps.to(weight_dtype),
                            batch_empty_text_embed,
                            class_labels=batch_task_emb,
                        ).sample
                    else:
                        lotus_pred = student_unet(
                            lotus_input,
                            lotus_timesteps.to(weight_dtype),
                            batch_empty_text_embed,
                            class_labels=batch_task_emb,
                        ).sample

                    # ---------------------------------
                    # decode pred_latent to depth
                    latent = lotus_pred / vae.config.scaling_factor
                    z = vae.post_quant_conv(latent.to(weight_dtype))
                    lotus_depth = vae.decoder(z).mean(dim=1, keepdim=True)

                    # ---------------------------------
                    # calculate difference
                    l1_error = torch.abs(lotus_depth - norm_disp_unidepth)
                    l1_error = l1_error / l1_error.max()
                    l1_error = l1_error.clip(0, 1)

                    latent_mask = torch.nn.functional.interpolate(l1_error, scale_factor=1 / 8)

                    noise = torch.randn_like(unidepth_latent)
                    noisy_lotus_latent = noise_scheduler.add_noise(
                        lotus_pred, noise, lotus_timesteps
                    )

                    if np.random.rand() < 0.8:
                        noisy_latent = noisy_lotus_latent * latent_mask + unidepth_latent * (
                            1 - latent_mask
                        )
                        student_input = torch.cat([rgb_latent, noisy_latent], dim=1)
                    else:
                        student_input = torch.cat([rgb_latent, noise], dim=1)

                # ---------------------------------
                pred_latent = student_unet(
                    student_input,
                    lotus_timesteps.to(weight_dtype),
                    encoder_hidden_states=batch_empty_text_embed,
                    class_labels=batch_task_emb,
                ).sample

                # ------------------------------------------------------------------
                # SDS loss
                noise = torch.randn_like(pred_latent)
                noisy_samples = noise_scheduler.add_noise(pred_latent, noise, lotus_timesteps)
                unet_input = torch.cat([rgb_latent.detach(), noisy_samples], dim=1).to(
                    weight_dtype
                )  # this order is important

                with torch.no_grad():
                    unet_pred = lotus_unet(
                        unet_input,
                        lotus_timesteps.to(weight_dtype),
                        batch_empty_text_embed,
                        class_labels=batch_task_emb,
                    ).sample

                sigma_t = ((1 - alphas_cumprod[lotus_timesteps]) ** 0.5).view(-1, 1, 1, 1)
                score_gradient = torch.nan_to_num(sigma_t**2 * (pred_latent - unet_pred))
                # ------------------------------------------------------------------

                # ------------------------------------------------------------------
                # Compute the SDS loss for the model
                target = (pred_latent - score_gradient).detach()
                sds_loss = 0.5 * F.mse_loss(pred_latent.float(), target.float(), reduction="mean")
                # ------------------------------------------------------------------

                # ---------------------------------
                # decode pred_latent to depth
                latent = pred_latent / vae.config.scaling_factor
                z = vae.post_quant_conv(latent.to(weight_dtype))
                pred_depth = vae.decoder(z).mean(dim=1, keepdim=True)

                # ------------------------------------------------------------------
                # Depth loss

                depth_loss = l1_loss(
                    pred_depth * 0.5 + 0.5, norm_disp_unidepth * 0.5 + 0.5, l1_error
                )

                # ------------------------------------------------------------------
                # Optimization
                loss = sds_loss + depth_loss * args.depth_weight
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_model.update()

                    params_to_clip = student_unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                # ------------------------------------------------------------------

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints) - args.checkpoints_total_limit + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            output_dir, removing_checkpoint
                                        )
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if global_step % args.validation_steps == 0 or global_step == 1:

                            student_unet.eval()
                            saved_dir = os.path.join(
                                output_dir, "visualization", f"iter_{global_step}"
                            )
                            os.makedirs(saved_dir, exist_ok=True)

                            pipeline = SharpDepthPipeline.from_pretrained(
                                student_ckpt_dir,
                                unet=unwrap_model(student_unet),
                                vae=unwrap_model(vae),
                                scheduler=noise_scheduler,
                            ).to(accelerator.device, dtype=weight_dtype)
                            with torch.no_grad():
                                for loader_idx, loader in enumerate(val_loaders):
                                    for vis_idx, batch in enumerate(loader):
                                        if vis_idx > 4:
                                            continue
                                        rgb = Image.fromarray(
                                            batch["rgb_int"]
                                            .squeeze()
                                            .permute(1, 2, 0)
                                            .cpu()
                                            .numpy()
                                            .astype(np.uint8)
                                        )
                                        out = pipeline(
                                            rgb, unidepth, processing_res=768, denoising_steps=1
                                        )

                                        depth_pred = torch.from_numpy(out.depth_np).to(
                                            accelerator.device
                                        )
                                        depth_uni = torch.from_numpy(out.depth_uni).to(
                                            accelerator.device
                                        )

                                        gt = (
                                            batch["depth_raw_linear"]
                                            .squeeze()
                                            .to(accelerator.device)
                                        )
                                        valid_mask = (
                                            batch["valid_mask_raw"].squeeze().to(accelerator.device)
                                        )

                                        error = abs_relative_difference_full(
                                            depth_pred, gt, valid_mask
                                        )
                                        error_uni = abs_relative_difference_full(
                                            depth_uni, gt, valid_mask
                                        )

                                        error_col = colorize(
                                            error.cpu().numpy(), 0, 0.12, cmap="coolwarm"
                                        )
                                        error_uni_col = colorize(
                                            error_uni.cpu().numpy(), 0, 0.12, cmap="coolwarm"
                                        )
                                        Image.fromarray(error_uni_col).save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_unidepth_error_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )
                                        Image.fromarray(error_col).save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_pred_depth_error_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )

                                        out.depth_colored.save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_pred_depth_{loader_idx}_{vis_idx}_{error.mean()}.jpg",
                                            )
                                        )
                                        out.unidepth_colored.save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_unidepth_{loader_idx}_{vis_idx}_{error_uni.mean()}.jpg",
                                            )
                                        )

                                        out.pred_mask.save(
                                            os.path.join(
                                                saved_dir,
                                                f"vis_diff_mask_{loader_idx}_{vis_idx}.jpg",
                                            )
                                        )
                                        rgb.save(
                                            os.path.join(
                                                saved_dir, f"vis_rgb_{loader_idx}_{vis_idx}.jpg"
                                            )
                                        )

                            del pipeline
                            torch.cuda.empty_cache()
                            student_unet.train()

                logs = {
                    "loss": loss.detach().item(),
                    "depth_loss": depth_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    break
