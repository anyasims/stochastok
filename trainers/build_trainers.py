"""
Builds the individual components of the trainer,
and the trainer itself.
"""

import os

import torch
from torch.distributed import init_process_group

from trainers.base_trainer import BaseTrainer
from trainers.dataset_interface_pretraining import BaseDatasetRandom
from trainers.dataset_interface_instruction import DatasetInterfaceInstruction
from trainers.loss_fn import (
    cross_entropy_loss_fn,
)
from trainers.optimizer import configure_nanoGPT_optimizer
from trainers.scheduler import (
    CosineLRScheduler,
    DropoutScheduler,
)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # Get the master address and port from SLURM environment variables
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    # Set the environment variables for PyTorch distributed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def build_trainer(cfg, model, gpu_id):
    """
    Given a config, this function builds a trainer
    and all relevant components of it.
    """

    # build optimizer
    optimizer = configure_nanoGPT_optimizer(
        model=model,
        weight_decay=cfg.trainer["optimizer"]["weight_decay"],
        learning_rate=cfg.trainer["optimizer"]["lr"],
        betas=(cfg.trainer["optimizer"]["beta1"], cfg.trainer["optimizer"]["beta2"]),
    )

    # build LR scheduler
    lr_scheduler = CosineLRScheduler(
        warmup_iters=cfg.trainer["training"]["warmup_iters"],
        decay_iters=cfg.trainer["training"]["lr_decay_iters"],
        lr=cfg.trainer["optimizer"]["lr"],
        min_lr=cfg.trainer["optimizer"]["min_lr"],
    )

    # build dropout scheduler
    dropout_scheduler = DropoutScheduler(cfg.trainer["dropout_scheduler"]["dropout"])

    # build dataloder
    if cfg.trainer["dataset"]["is_instruction"]:
        train_dataset = DatasetInterfaceInstruction(cfg=cfg, split="train")
        val_dataset = DatasetInterfaceInstruction(cfg=cfg, split="val")
    else:
        train_dataset = BaseDatasetRandom(cfg=cfg, split="train")
        val_dataset = BaseDatasetRandom(cfg=cfg, split="val")


    # wrap in dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg["trainer"]["training"]["batch_size"],
        shuffle=False,

    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg["trainer"]["training"]["batch_size"],
        shuffle=False,
    )

    # build loss function
    loss_fn = cross_entropy_loss_fn

    # build the trainer
    trainer = BaseTrainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dropout_scheduler=dropout_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        gpu_id=gpu_id,
    )

    return trainer
