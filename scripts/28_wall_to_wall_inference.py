"""Phase 5 step 1: patch-based wall-to-wall biomass inference across the dev AOI.

Fully convolutional inference was abandoned because the trained model's
AdaptiveAvgPool2d + strided downsampling is not translation-equivariant: a large
tile's interior pixels do not reproduce the patch-based prediction. Instead we do
correct patch-based inference on a stride-10 grid, which reproduces the training
computation exactly (verified: reproduces Phase 4 test predictions to <0.2 Mg/ha,
correlation 1.0000).

For speed, patches are not read one at a time. The AOI is processed in horizontal
strips: each strip reads one padded row-block of all three rasters into memory with
a few large sequential reads, then slices every stride-10 patch in that strip from
memory. This turns millions of tiny windowed reads into ~100 large reads per job.

Output: one float32 GeoTIFF per (variant, seed) at 100 m resolution (stride-10 on
the 10 m grid), saved under data/processed/maps/. This matches ESA CCI Biomass v5
resolution, simplifying the Phase 5 validation comparison.

The 6 default jobs:
    - biomass_late_fusion_seed42, seed7, seed123
    - biomass_s2_only_seed7, biomass_s1_only_seed7, biomass_early_fusion_seed7

Input year: 2021.

Usage:
    uv run python scripts/28_wall_to_wall_inference.py
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import rasterio
import torch
from affine import Affine
from omegaconf import DictConfig
from rasterio.windows import Window
from torch.amp import autocast

from biomass.config import INTERIM_DIR, PROCESSED_DIR
from biomass.log_setup import setup_logging
from biomass.models.variants import build_model
from biomass.training.dataset import Variant

log = logging.getLogger(__name__)

# -------------------- Configuration --------------------

CHECKPOINT_DIR = Path("data/checkpoints")
OUTPUT_DIR = PROCESSED_DIR / "maps"
STATE_FILE = INTERIM_DIR / "wall_to_wall_state.json"

INPUT_YEAR = 2021
PATCH_HALF = 12          # 25x25 patches (matches training)
PATCH_SIZE = 25
STRIDE = 10              # output grid stride in 10 m pixels -> 100 m output resolution
BATCH_SIZE = 256
AGBD_CLIP_MAX = 500.0    # training label cap; clip extreme extrapolations
ROWS_PER_STRIP = 40      # output rows per strip (40 * STRIDE = 400 raster rows/strip)

# Channel indices into the full 15-channel stack, per variant.
# Full order: S2 [0..9], S1 [10..12], DEM [13, 14].
VARIANT_CHANNEL_INDICES = {
    Variant.S1_ONLY: [10, 11, 12, 13, 14],
    Variant.S2_ONLY: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14],
    Variant.EARLY: list(range(15)),
    Variant.LATE: list(range(15)),
}


@dataclass
class InferenceJob:
    variant: Variant
    seed: int
    run_name: str
    output_name: str


def default_jobs() -> list[InferenceJob]:
    jobs = []
    for seed in (42, 7, 123):
        jobs.append(InferenceJob(Variant.LATE, seed, f"late_fusion_seed{seed}",
                                 f"biomass_late_fusion_seed{seed}.tif"))
    for vstr, venum, prefix in [
        ("s2_only", Variant.S2_ONLY, "s2_only"),
        ("s1_only", Variant.S1_ONLY, "s1_only"),
        ("early_fusion", Variant.EARLY, "early_fusion"),
    ]:
        jobs.append(InferenceJob(venum, 7, f"{vstr}_seed7", f"biomass_{prefix}_seed7.tif"))
    return jobs


# -------------------- Raster IO --------------------

class RasterStack:
    """Opens S2 + S1 + DEM rasters for a year and reads normalized row-blocks.

    Normalization uses the patches_dev_meta.json per-channel statistics, matching
    the training dataset exactly.
    """

    def __init__(self, aoi_name: str, year: int):
        s2_path = PROCESSED_DIR / f"s2_composite_{aoi_name}_{year}.tif"
        s1_path = PROCESSED_DIR / f"s1_composite_{aoi_name}_{year}.tif"
        dem_path = PROCESSED_DIR / f"dem_{aoi_name}.tif"
        meta_path = PROCESSED_DIR / f"patches_{aoi_name}_meta.json"
        for p in (s2_path, s1_path, dem_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"Required input missing: {p}")

        self.s2 = rasterio.open(s2_path)
        self.s1 = rasterio.open(s1_path)
        self.dem = rasterio.open(dem_path)
        self.width = self.s2.width
        self.height = self.s2.height
        self.transform = self.s2.transform
        self.crs = self.s2.crs

        meta = json.load(open(meta_path))
        names = meta["channels"]
        norm = meta["normalization"]
        self.mean = np.array([norm[n]["mean"] for n in names], dtype=np.float32)
        self.std = np.array([norm[n]["std"] for n in names], dtype=np.float32)
        assert len(self.mean) == 15, f"Expected 15 channel stats, got {len(self.mean)}"

        log.info(f"RasterStack: {self.width}x{self.height} at {self.crs}, year {year}")
        log.info(f"  S2:  {s2_path.name}")
        log.info(f"  S1:  {s1_path.name}")
        log.info(f"  DEM: {dem_path.name}")

    def read_row_block(self, row_start: int, row_end: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Read a full-width block of rows [row_start, row_end), padded by PATCH_HALF
        on all sides so any patch centered in the block has full context.

        Returns (block, finite, block_row_origin, block_col_origin):
          block: (15, H_padded, W_padded) float32, normalized, NaN->0
          finite: (15, H_padded, W_padded) bool, True where raw pixel was finite
          block_row_origin: raster row corresponding to block[:, 0, :]
          block_col_origin: raster col corresponding to block[:, :, 0]
        """
        pad = PATCH_HALF
        r0 = row_start - pad
        c0 = -pad
        h = (row_end + pad) - r0
        w = (self.width + pad) - c0
        win = Window(col_off=c0, row_off=r0, width=w, height=h)

        s2 = self.s2.read(window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        s1 = self.s1.read(window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        dem = self.dem.read(window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        raw = np.concatenate([s2, s1, dem], axis=0)  # (15, H_padded, W_padded)

        finite = np.isfinite(raw)
        normd = (raw - self.mean[:, None, None]) / self.std[:, None, None]
        normd = np.nan_to_num(normd, nan=0.0).astype(np.float32)
        return normd, finite, r0, c0

    def close(self):
        self.s2.close()
        self.s1.close()
        self.dem.close()


# -------------------- Grid --------------------

def build_output_grid(width: int, height: int, stride: int):
    """Stride-spaced patch-center pixels. Returns (row_centers, col_centers, out_h, out_w)."""
    row_centers = np.arange(PATCH_HALF, height - PATCH_HALF, stride, dtype=np.int64)
    col_centers = np.arange(PATCH_HALF, width - PATCH_HALF, stride, dtype=np.int64)
    return row_centers, col_centers, len(row_centers), len(col_centers)


# -------------------- Inference --------------------

@torch.no_grad()
def _run_batch(model, patches, coords, valids, channel_idx, output, device):
    """Run one batch of patches through the model, scatter results into output array."""
    arr = np.stack(patches, axis=0)          # (B, 15, 25, 25)
    arr = arr[:, channel_idx]                # slice to variant channels
    x = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        preds = model(x)
    preds = preds.float().cpu().numpy().reshape(-1)
    preds = np.clip(preds, 0.0, AGBD_CLIP_MAX)
    for k, (oi, oj) in enumerate(coords):
        if valids[k]:
            output[oi, oj] = preds[k]


@torch.no_grad()
def run_job(job: InferenceJob, stack: RasterStack, cfg: DictConfig, device: torch.device) -> None:
    ckpt_path = CHECKPOINT_DIR / job.run_name / "best.pt"
    out_path = OUTPUT_DIR / job.output_name
    if out_path.exists():
        log.info(f"  Output exists ({out_path.name}), skipping.")
        return

    model = build_model(variant=job.variant,
                        hidden_dim=cfg.train.head_hidden_dim,
                        dropout=cfg.train.head_dropout).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    channel_idx = np.array(VARIANT_CHANNEL_INDICES[job.variant])
    log.info(f"  Variant {job.variant.value}: channels {list(channel_idx)}")

    row_centers, col_centers, out_h, out_w = build_output_grid(
        stack.width, stack.height, STRIDE)
    log.info(f"  Output grid: {out_h} x {out_w} = {out_h*out_w:,} patches "
             f"(stride {STRIDE} -> {STRIDE*10} m resolution)")

    output = np.full((out_h, out_w), -9999.0, dtype=np.float32)
    pad = PATCH_HALF
    t0 = time.time()
    n_done = 0
    n_total = out_h * out_w

    for strip_start in range(0, out_h, ROWS_PER_STRIP):
        strip_end = min(strip_start + ROWS_PER_STRIP, out_h)
        raster_row_start = int(row_centers[strip_start])
        raster_row_end = int(row_centers[strip_end - 1]) + 1

        block, finite, block_row_origin, block_col_origin = stack.read_row_block(
            raster_row_start, raster_row_end)

        patches, coords, valids = [], [], []
        for oi in range(strip_start, strip_end):
            rc = int(row_centers[oi])
            br = rc - block_row_origin  # center row within block
            for oj in range(out_w):
                cc = int(col_centers[oj])
                bc = cc - block_col_origin  # center col within block
                patch = block[:, br - pad: br + pad + 1, bc - pad: bc + pad + 1]
                center_ok = bool(finite[:, br, bc].all())
                patches.append(patch)
                coords.append((oi, oj))
                valids.append(center_ok)

                if len(patches) >= BATCH_SIZE:
                    _run_batch(model, patches, coords, valids, channel_idx, output, device)
                    n_done += len(patches)
                    patches, coords, valids = [], [], []

        if patches:
            _run_batch(model, patches, coords, valids, channel_idx, output, device)
            n_done += len(patches)

        elapsed = time.time() - t0
        eta = elapsed / max(n_done, 1) * (n_total - n_done)
        log.info(f"    rows {strip_end}/{out_h}  "
                 f"({n_done:,}/{n_total:,}, {n_done/n_total*100:.1f}%)  "
                 f"elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min")

    coverage = (output != -9999.0).mean() * 100
    log.info(f"  Coverage {coverage:.1f}%")

    src_t = stack.transform
    new_transform = Affine(
        src_t.a * STRIDE, src_t.b, src_t.c + src_t.a * PATCH_HALF,
        src_t.d, src_t.e * STRIDE, src_t.f + src_t.e * PATCH_HALF,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": out_h, "width": out_w, "count": 1,
        "dtype": "float32", "crs": stack.crs, "transform": new_transform,
        "nodata": -9999.0, "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256, "predictor": 3,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(output, 1)
    log.info(f"  Wrote {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -------------------- State + main --------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.load(open(STATE_FILE))
    return {"completed": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(state, open(STATE_FILE, "w"), indent=2)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("=" * 70)
    log.info("Phase 5 step 1: patch-based wall-to-wall inference (strip-batched)")
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    stack = RasterStack(cfg.aoi.name, INPUT_YEAR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    jobs = default_jobs()
    log.info(f"{len(jobs)} jobs; {len(state['completed'])} already completed")

    for i, job in enumerate(jobs):
        log.info("")
        log.info(f"=== Job {i+1}/{len(jobs)}: {job.run_name} ===")
        if job.output_name in state["completed"]:
            log.info("  Already completed, skipping.")
            continue
        try:
            run_job(job, stack, cfg, device)
            state["completed"].append(job.output_name)
            save_state(state)
        except Exception as e:
            log.error(f"  Job failed: {e}", exc_info=True)

    stack.close()
    log.info("")
    log.info(f"Done. Completed {len(state['completed'])}/{len(jobs)}")


if __name__ == "__main__":
    main()
    