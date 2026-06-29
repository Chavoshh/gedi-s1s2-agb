"""Phase 4 step 1: run inference on the test partition for all 12 checkpoints.

Loads each (variant, seed) checkpoint, runs inference on the 72,033 test
patches (no augmentation, no dropout, eval mode), and saves predictions to
a single parquet file with columns: patch_id, variant, seed, true_agbd,
pred_agbd. Predictions are clipped at 0 (no negative biomass).

This is the expensive step (~3-5 hours total). Script 26 (cheap) consumes the
parquet to compute all metrics and figures.

Usage:
    uv run python scripts/25_run_inference.py
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.amp import autocast
from torch.utils.data import DataLoader

from biomass.config import PROCESSED_DIR
from biomass.log_setup import setup_logging
from biomass.models.variants import build_model
from biomass.training.dataset import BiomassPatchDataset, Variant

log = logging.getLogger(__name__)

VARIANTS = ["s2_only", "s1_only", "early", "late"]
SEEDS = [42, 7, 123]
CHECKPOINT_DIR = Path("data/checkpoints")
OUTPUT_PATH = PROCESSED_DIR / "test_predictions.parquet"


def run_name(variant: str, seed: int) -> str:
    """Match the naming used in scripts/21_train.py."""
    name_map = {
        "s2_only": "s2_only",
        "s1_only": "s1_only",
        "early": "early_fusion",
        "late": "late_fusion",
    }
    return f"{name_map[variant]}_seed{seed}"


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference; return (predictions, ground_truth) as float32 arrays."""
    model.eval()
    preds_list = []
    truth_list = []
    for patch, label in loader:
        patch = patch.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            pred = model(patch)
        # Clip negatives at metric/save time only; keep raw model output for now
        # so script 26 can apply clipping consistently with the training metrics.
        preds_list.append(pred.float().cpu().numpy())
        truth_list.append(label.numpy())
    return np.concatenate(preds_list), np.concatenate(truth_list)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("=" * 70)
    log.info("Phase 4 step 1: test-set inference for 12 checkpoints")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    zarr_path = PROCESSED_DIR / f"patches_{cfg.aoi.name}.zarr"
    meta_path = PROCESSED_DIR / f"patches_{cfg.aoi.name}_meta.json"

    all_records = []
    t_start = time.time()

    for variant_str in VARIANTS:
        variant = Variant(variant_str)
        log.info("")
        log.info(f"--- Variant: {variant_str} ---")

        # Build the test dataset and loader once per variant (DataLoader is reusable
        # across seeds because no augmentation and a fixed test split).
        test_ds = BiomassPatchDataset(
            zarr_path=zarr_path,
            meta_path=meta_path,
            split="test",
            variant=variant,
            augment=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=128,           # larger than training; no gradients to store
            shuffle=False,            # IMPORTANT: preserves index order for joining
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
        )

        # Patch IDs in the order the test loader will produce them.
        # The dataset's self.indices array gives the global patch IDs in test order.
        test_patch_ids = test_ds.indices.copy()

        for seed in SEEDS:
            name = run_name(variant_str, seed)
            ckpt_path = CHECKPOINT_DIR / name / "best.pt"
            if not ckpt_path.exists():
                log.error(f"  Missing checkpoint: {ckpt_path}. Skipping.")
                continue

            log.info(f"  Loading {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            model = build_model(
                variant=variant,
                hidden_dim=cfg.train.head_hidden_dim,
                dropout=cfg.train.head_dropout,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])

            t_inf = time.time()
            preds, truth = run_inference(model, test_loader, device)
            elapsed = time.time() - t_inf
            log.info(
                f"  {name}: {len(preds):,} predictions in {elapsed/60:.1f} min "
                f"({len(preds)/elapsed:.0f} patches/sec)"
            )

            # Sanity checks
            if len(preds) != len(test_patch_ids):
                raise RuntimeError(
                    f"Mismatch: {len(preds)} predictions vs {len(test_patch_ids)} test patches"
                )

            all_records.append(pd.DataFrame({
                "patch_id": test_patch_ids,
                "variant": variant_str,
                "seed": seed,
                "true_agbd": truth.astype(np.float32),
                "pred_agbd": preds.astype(np.float32),
            }))

            # Free GPU memory before the next checkpoint
            del model
            torch.cuda.empty_cache()

    # Combine and save
    df = pd.concat(all_records, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    total_elapsed = (time.time() - t_start) / 60
    log.info("")
    log.info("=" * 70)
    log.info(f"Inference complete in {total_elapsed:.1f} min")
    log.info(f"  {len(df):,} prediction rows (expected {12 * 72033:,})")
    log.info(f"  Saved to {OUTPUT_PATH}")
    log.info(f"  File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")
    log.info("=" * 70)


if __name__ == "__main__":
    main()