"""Phase 3, training entry point.

Trains one model variant for one seed with Hydra-configured hyperparameters.
Logs to W&B, saves best checkpoint by val RMSE, supports early stopping.

Usage:
    # Train S2-only baseline with default config
    uv run python scripts/21_train.py

    # Train another variant
    uv run python scripts/21_train.py model=s1_only

    # Override training hyperparameters
    uv run python scripts/21_train.py model=late train.batch_size=32 train.epochs=30

    # Change seed
    uv run python scripts/21_train.py model=early train.seed=123
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from biomass.config import PROCESSED_DIR
from biomass.log_setup import setup_logging
from biomass.models.variants import build_model, count_parameters
from biomass.training.dataset import BiomassPatchDataset, Variant
from biomass.training.loop import train_one_epoch, validate
from biomass.training.losses import HuberLoss

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    lr_min: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine annealing to lr_min."""
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=lr_min)
    warmup = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=lr_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_rmse: float,
    cfg: DictConfig,
) -> None:
    """Save a training checkpoint and the resolved config alongside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_rmse": val_rmse,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
    log.info(f"  Checkpoint saved: {path.name} (val_rmse={val_rmse:.3f})")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    # ---- Setup ----
    setup_logging(cfg.log_level)
    log.info("=" * 70)
    log.info(f"Phase 3 training: variant={cfg.model.variant}, seed={cfg.train.seed}")
    log.info("=" * 70)
    log.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    variant = Variant(cfg.model.variant)
    run_name = f"{cfg.model.name}_seed{cfg.train.seed}"

    # ---- Data ----
    zarr_path = PROCESSED_DIR / f"patches_{cfg.aoi.name}.zarr"
    meta_path = PROCESSED_DIR / f"patches_{cfg.aoi.name}_meta.json"
    log.info(f"Loading dataset from {zarr_path}")

    train_ds = BiomassPatchDataset(
        zarr_path=zarr_path, meta_path=meta_path,
        split="train", variant=variant, augment=True,
    )
    val_ds = BiomassPatchDataset(
        zarr_path=zarr_path, meta_path=meta_path,
        split="val", variant=variant, augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
    )
    log.info(
        f"Loaders ready: train batches={len(train_loader):,}, "
        f"val batches={len(val_loader):,}, batch_size={cfg.train.batch_size}"
    )

    # ---- Model ----
    model = build_model(
        variant=variant,
        hidden_dim=cfg.train.head_hidden_dim,
        dropout=cfg.train.head_dropout,
    ).to(device)
    n_params = count_parameters(model)
    log.info(f"Model built: variant={variant.value}, params={n_params/1e6:.2f}M")

    # ---- Loss / optimizer / scheduler / AMP ----
    loss_fn = HuberLoss(delta=cfg.train.huber_delta).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.epochs,
        lr_min=cfg.train.lr_min,
    )
    scaler = GradScaler(device="cuda") if cfg.train.use_amp else None

    # ---- W&B ----
    wandb.init(
        project=cfg.train.wandb_project,
        name=run_name,
        mode=cfg.train.wandb_mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[variant.value, f"seed{cfg.train.seed}", cfg.aoi.name],
    )

    # ---- Training loop ----
    ckpt_dir = Path(cfg.train.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_rmse = float("inf")
    epochs_since_improvement = 0
    t_start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        t_epoch = time.time()
        log.info(f"\n--- Epoch {epoch}/{cfg.train.epochs} ---")

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device,
            log_every=cfg.train.log_every_n_batches,
        )
        val_metrics = validate(
            model, val_loader, loss_fn, device, use_amp=cfg.train.use_amp,
        )
        scheduler.step()

        epoch_time = time.time() - t_epoch
        current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch} ({epoch_time/60:.1f} min): "
            f"train loss={train_metrics['loss']:.3f} RMSE={train_metrics['rmse']:.2f} | "
            f"val loss={val_metrics['loss']:.3f} RMSE={val_metrics['rmse']:.2f} "
            f"R^2={val_metrics['r2']:.3f} bias={val_metrics['bias']:+.2f} | "
            f"lr={current_lr:.2e}"
        )

        wandb.log({
            "epoch": epoch,
            "epoch_time_min": epoch_time / 60,
            "learning_rate": current_lr,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        })

        # Checkpoint + early stopping
        improved = val_metrics["rmse"] < best_val_rmse - cfg.train.early_stop_min_delta
        if improved:
            best_val_rmse = val_metrics["rmse"]
            epochs_since_improvement = 0
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scheduler, epoch,
                best_val_rmse, cfg,
            )
            wandb.run.summary["best_val_rmse"] = best_val_rmse
            wandb.run.summary["best_epoch"] = epoch
        else:
            epochs_since_improvement += 1
            log.info(f"  No improvement ({epochs_since_improvement}/{cfg.train.early_stop_patience})")

        if epochs_since_improvement >= cfg.train.early_stop_patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    # Always save the last checkpoint too
    save_checkpoint(
        ckpt_dir / "last.pt",
        model, optimizer, scheduler, epoch,
        val_metrics["rmse"], cfg,
    )

    total_time = (time.time() - t_start) / 60
    log.info(
        f"\nTraining done in {total_time:.1f} min. "
        f"Best val RMSE: {best_val_rmse:.3f} Mg/ha"
    )
    wandb.run.summary["total_time_min"] = total_time
    wandb.finish()


if __name__ == "__main__":
    main()