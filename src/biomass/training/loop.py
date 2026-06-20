"""Training and validation loops.

Single-epoch primitives that the main training script composes. Designed to
be model-agnostic: any nn.Module that maps (B, C, H, W) -> (B,) works.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from biomass.training.metrics import StreamingRegressionMetrics

log = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    log_every: int = 50,
) -> dict[str, float]:
    """Train the model for one epoch. Returns train metrics + average loss.

    If scaler is provided, uses AMP mixed precision. Otherwise full FP32.
    """
    model.train()
    metrics = StreamingRegressionMetrics()
    loss_sum = 0.0
    n_batches = 0

    for batch_idx, (patch, label) in enumerate(loader):
        patch = patch.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type="cuda", dtype=torch.float16):
                pred = model(patch)
                loss = loss_fn(pred, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(patch)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.item())
        n_batches += 1
        metrics.update(pred.detach(), label.detach())

        if (batch_idx + 1) % log_every == 0:
            log.info(
                f"  train batch {batch_idx + 1}/{len(loader)}: "
                f"loss={loss.item():.3f}"
            )

    results = metrics.compute()
    results["loss"] = loss_sum / max(n_batches, 1)
    return results


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """Evaluate on the validation loader. Returns val metrics + average loss."""
    model.eval()
    metrics = StreamingRegressionMetrics()
    loss_sum = 0.0
    n_batches = 0

    for patch, label in loader:
        patch = patch.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                pred = model(patch)
                loss = loss_fn(pred, label)
        else:
            pred = model(patch)
            loss = loss_fn(pred, label)

        loss_sum += float(loss.item())
        n_batches += 1
        metrics.update(pred, label)

    results = metrics.compute()
    results["loss"] = loss_sum / max(n_batches, 1)
    return results