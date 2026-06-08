# gedi-s1s2-agb

Above-ground biomass regression from GEDI L4A footprints fused with Sentinel-1 SAR and Sentinel-2 optical imagery, producing continuous wall-to-wall biomass estimates over forested regions.

**Author:** Chavosh Almassian — M.Sc. Remote Sensing & Geoinformatics, Karlsruhe Institute of Technology
**Status:** Phase 1 — GEDI data acquisition & filtering
**Target venue:** IEEE GRSL / JSTARS

## Project overview

GEDI (Global Ecosystem Dynamics Investigation) is a NASA spaceborne lidar that provides footprint-level biomass measurements but only as sparse along-track samples. This project uses those samples as supervision for a deep learning model that maps Sentinel-1/2 features to biomass, enabling dense biomass maps from the resulting model.

The work compares early- and late-fusion strategies for combining SAR and optical inputs, with a focus on rigorous spatial cross-validation, uncertainty quantification, and saturation analysis.

## Study area

Two AOIs (Areas Of Interest) are defined:

- **`dev`** — MGRS tile 29TNG, central Galicia (~110×110 km). Used for prototyping the full pipeline end-to-end.
- **`full`** — Northwest Iberia (lon −9.5° to −5.5°, lat 41.5° to 43.5°). Used for the final paper results.

## Hardware target

- Windows 11, Python 3.11, uv-managed environment
- NVIDIA GTX 1050 Ti (4 GB VRAM, CC 6.1)
- 16 GB RAM, Core i3

The pipeline is designed to fit within these constraints via patch-based regression, mixed-precision training, and streamed inference.

## Setup

### 1. Install uv and Python 3.11

```powershell
# Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
# Then in the project directory:
uv sync --extra dev
```

### 2. Install PyTorch with CUDA (separate step, custom index)

```powershell
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

We pin torch outside `pyproject.toml` because the CUDA-enabled wheel requires a custom index URL.

### 3. Credentials

Copy `.env.example` to `.env` and fill in:

- An NASA Earthdata Login account (free, from <https://urs.earthdata.nasa.gov/>)
- A Weights & Biases account (free, from <https://wandb.ai/>)

The first time you run any GEDI script, persist your Earthdata credentials to `_netrc`:

```powershell
uv run python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
```

### 4. Install optional dependency groups as needed

```powershell
# When you start working with Sentinel imagery
uv sync --extra sentinel

# When you start training models
uv sync --extra model

# Everything at once
uv sync --extra all
```

## Pipeline phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ done | Scoping, AOI selection, project scaffolding |
| 1 | 🚧 in progress | GEDI L4A acquisition and quality filtering |
| 2 | pending | Sentinel-1/2 co-location and feature extraction |
| 3 | pending | Tabular baseline (LightGBM) |
| 4 | pending | Deep learning models (S1-only, S2-only, early/late fusion) |
| 5 | pending | Wall-to-wall inference |
| 6 | pending | Evaluation: spatial CV, saturation analysis, ablations |
| 7 | pending | Uncertainty quantification |
| 8 | pending | Manuscript |

## Project layout

```
.
├── pyproject.toml         # Project metadata, dependencies, tool config
├── uv.lock                # Locked dependency versions (committed)
├── .env.example           # Credential template
├── README.md
├── LICENSE                # MIT
│
├── src/biomass/           # Importable package
│   ├── config.py          # Constants: beams, variables, paths
│   ├── data/
│   │   ├── aoi.py         # Named AOIs (dev, full)
│   │   └── gedi.py        # GEDI L4A read and filter
│   ├── logging.py         # Logging setup
│   └── ...
│
├── configs/               # Hydra configs
│   ├── base.yaml
│   └── aoi/
│       ├── dev.yaml
│       └── full.yaml
│
├── scripts/               # CLI entry points (numbered by phase)
│   ├── 01_query_gedi.py
│   ├── 02_prototype_granule_read.py
│   └── 03_extract_all_shots.py
│
├── tests/
│   └── test_imports.py
│
├── docs/
│   └── decisions.md       # Decision log (publication methods notes)
│
├── data/                  # gitignored
│   ├── raw/
│   ├── interim/
│   └── processed/
│
└── notebooks/             # Exploration only; not committed if scratch
```

## Reproducibility

- All dependency versions locked in `uv.lock` (committed).
- All design decisions recorded in `docs/decisions.md` with date and rationale.
- All scripts are Hydra-configurable; configs are versioned.
- Random seeds documented in each model config.

## License

MIT — see `LICENSE`.

## Contact

Chavosh Almassian
- chavosh@outlook.com
- [LinkedIn](https://www.linkedin.com/in/chavosh-almassian-81a05216a/)
- [GitHub](https://github.com/Chavoshh)