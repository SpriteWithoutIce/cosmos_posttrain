#!/usr/bin/env python3
"""
Training script for Joint Video-Action Generation.
Self-contained, no external cosmos-predict2.5 dependency.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.cosmost.models import JointVideoActionModel
from src.cosmost.datasets import LeRobotLatentDataset, collate_fn


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def get_dataloader(config, rank, world_size, split="train"):
    """Create dataloader."""
    dataset = LeRobotLatentDataset(
        lerobot_root=config["lerobot_root"],
        latent_root=config["latent_root"],
        num_cond_frames=config.get("num_cond_frames", 4),
        num_pred_frames=config.get("num_pred_frames", 8),
        num_actions_per_frame=config.get("num_actions_per_frame", 8),
        time_division_factor=config.get("time_division_factor", 4),
        action_dim=config.get("action_dim", 16),
        data_split=split,
        normalize_action=config.get("normalize_action", True),
        global_stats_json=config.get("global_stats_json"),
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(split=="train")) if world_size > 1 else None
    
    batch_size = config.get("batch_size", 1)
    
    # Check if dataset is smaller than batch_size
    if len(dataset) < batch_size * world_size:
        print(f"[Warning] Dataset size ({len(dataset)}) is smaller than batch_size ({batch_size}) * world_size ({world_size})")
        print(f"[Warning] Consider reducing batch_size")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and split=="train"),
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,  # Changed to False to avoid empty dataloader
    )
    
    return dataloader, sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.py")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.suffix == ".py":
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.get_config()
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Print config for debugging
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        print("=" * 50)
        print("Configuration:")
        print(f"  Lerobot root: {config.get('lerobot_root')}")
        print(f"  Latent root: {config.get('latent_root')}")
        print(f"  Batch size: {config.get('batch_size')}")
        print(f"  Num workers: {config.get('num_workers')}")
        print("=" * 50)
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Create model
    model = JointVideoActionModel(
        video_dit_config=config["video_dit"],
        action_head_enabled=config.get("action_head_enabled", True),
        action_head_cfg=config.get("action_head"),
        action_loss_weight=config.get("action_loss_weight", 1.0),
        action_stop_gradient=config.get("action_stop_gradient", False),
        num_conditional_frames=config.get("num_cond_frames", 4),
    ).cuda()
    
    # Use bfloat16 for mixed precision training (saves ~50% memory)
    use_amp = config.get("use_amp", True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=use_amp)
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        betas=(0.9, 0.99),
    )
    
    # Resume
    start_iter = 0
    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        start_iter = model.module.load_checkpoint(args.resume, optimizer) if hasattr(model, "module") else model.load_checkpoint(args.resume, optimizer)
    
    # Dataloader
    train_loader, train_sampler = get_dataloader(config, rank, world_size, "train")
    
    # Check if dataset is empty
    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty! Check your data paths.")
    
    if is_main:
        print(f"Dataset size: {len(train_loader.dataset)}, Batches per epoch: {len(train_loader)}")
    
    # Logging
    writer = SummaryWriter(output_dir / "logs") if is_main else None
    
    # Training loop
    model.train()
    iteration = start_iter
    max_iterations = config.get("max_iterations", 100000)
    save_every = config.get("save_every", 5000)
    log_every = config.get("log_every", 100)
    epoch = 0
    
    while iteration < max_iterations:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        for batch_idx, data_batch in enumerate(train_loader):
            if iteration >= max_iterations:
                break
            
            # Move to GPU
            for key in data_batch:
                if torch.is_tensor(data_batch[key]):
                    data_batch[key] = data_batch[key].cuda()
            
            # Forward with mixed precision
            model_module = model.module if hasattr(model, "module") else model
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                log_dict, loss = model_module.training_step(data_batch, iteration)
            
            # Backward with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            if is_main:
                if iteration % log_every == 0:
                    for key, value in log_dict.items():
                        writer.add_scalar(f"train/{key}", value.item() if torch.is_tensor(value) else value, iteration)
                
                if iteration % save_every == 0:
                    save_path = output_dir / "checkpoints" / f"checkpoint_{iteration:06d}.pt"
                    model_module.save_checkpoint(save_path, optimizer, iteration)
            
            iteration += 1
        
        epoch += 1
    
    if is_main:
        writer.close()
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
