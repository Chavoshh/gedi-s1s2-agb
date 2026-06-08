# Early vs. Late Fusion of Sentinel-1 and Sentinel-2 for GEDI-Supervised Biomass Regression

> Sparse-supervised regression for above-ground biomass: GEDI L4A lidar footprints as labels, Sentinel-1 SAR and Sentinel-2 optical imagery as inputs, dense biomass maps as outputs.

**Status:** Phase 1 — GEDI data acquisition & filtering
**Target venue:** IEEE GRSL / JSTARS
**Author:** Chavosh Almassian, M.Sc. Remote Sensing & Geoinformatics, Karlsruhe Institute of Technology
**Decisions log:** [`docs/decisions.md`](docs/decisions.md)

---

## Method at a glance

GEDI (Global Ecosystem Dynamics Investigation) is a NASA spaceborne lidar that measures above-ground biomass density (AGBD) at the scale of ~25 m footprints, sampled along the ISS orbit track. Its measurements are accurate but sparse — they cover roughly 4% of the land surface within its ±51.6° latitude band. This project uses those footprints as supervision for a model that maps wall-to-wall Sentinel-1 SAR and Sentinel-2 optical features to biomass, allowing biomass to be predicted at every pixel across an Area Of Interest (AOI).

```
                 Sparse supervision                Dense input
                 ─────────────────                 ───────────
                 GEDI L4A footprints       ╲      Sentinel-1 (VV, VH)
                 (~25 m, sparse points)     ╲     Sentinel-2 (10 bands)
                          │                  ╲    Copernicus DEM
                          │                   ▼   (10 m, wall-to-wall)
                          │
                          ▼
                  ┌────────────────────────────────┐
                  │  Patch-based regression model  │
                  │  S1-only / S2-only / early /   │
                  │       late fusion variants     │
                  └────────────────────────────────┘
                          │
                          ▼
                 Wall-to-wall AGBD map
                 (Mg/ha, with per-pixel uncertainty)
```

The methodological focus is a controlled comparison of SAR-only, optical-only, early-fusion, and late-fusion variants of the same backbone, evaluated under strict spatial cross-validation with explicit saturation analysis and per-pixel uncertainty estimation.

---

## Study area

Two AOIs are defined in [`configs/aoi/`](configs/aoi/), swappable at the command line via Hydra.

| AOI    | Coverage                                              | Size            | Purpose             |
|--------|-------------------------------------------------------|-----------------|---------------------|
| `dev`  | MGRS tile 29TNG, central Galicia                      | ~110 × 110 km   | Pipeline prototyping |
| `full` | Northwest Iberia (lon −9.5° to −5.5°, lat 41.5° to 43.5°) | ~400 × 220 km   | Final paper results  |

Northwest Iberia was chosen for the breadth of its biomass dynamic range (dense Atlantic forest, Mediterranean oak, dehesa savanna), favorable Sentinel-2 cloud climatology relative to central Europe, GEDI shot density at its latitude, and the availability of reference data (Spanish IFN, ESA CCI Biomass). Full rationale in the decisions log.

---

## Pipeline phases

| Phase | Status | Description |
|------|------|------|
| 0 | ✅ done | Scoping, AOI selection, project scaffolding, GEDI access validated |
| 1 | 🚧 in progress | GEDI L4A acquisition and quality filtering |
| 2 | pending | Sentinel-1/2 co-location and patch extraction |
| 3 | pending | Tabular baseline (LightGBM on summary features) |
| 4 | pending | Deep learning models (S1-only, S2-only, early-fusion, late-fusion) |
| 5 | pending | Wall-to-wall inference and CCI Biomass comparison |
| 6 | pending | Evaluation: spatial CV, saturation analysis, ablations |
| 7 | pending | Uncertainty quantification |
| 8 | pending | Manuscript |

---

## Data sources

| Product                        | Resolution | Role                             | Provider           |
|--------------------------------|-----------|----------------------------------|--------------------|
| GEDI L4A V2.1 footprint AGBD   | 25 m      | Sparse supervision labels        | NASA ORNL DAAC     |
| Sentinel-1 GRD (RTC)           | 10 m      | SAR backscatter (VV, VH)         | ESA / Microsoft PC |
| Sentinel-2 L2A                 | 10–20 m   | Optical reflectance              | ESA / Microsoft PC |
| Copernicus DEM GLO-30          | 30 m      | Topography (auxiliary input)     | ESA                |
| ESA WorldCover 2021            | 10 m      | Forest mask for inference        | ESA                |
| ESA CCI Biomass v5             | 100 m     | Independent comparison product   | ESA                |

---

## Setup

### 1. Environment

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
uv sync --extra dev
```

This installs the core data-handling stack and dev tools. The `sentinel`, `model`, and `viz` dependency groups are installed on demand later:

```powershell
uv sync --extra sentinel   # Sentinel imagery and STAC
uv sync --extra model      # LightGBM, W&B (PyTorch is installed separately)
uv sync --extra viz        # matplotlib, seaborn, contextily
uv sync --extra all        # everything at once
```

### 2. PyTorch with CUDA

PyTorch is not in `pyproject.toml` because the CUDA wheel needs a custom index URL:

```powershell
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Credentials

You need a free NASA Earthdata Login account (from <https://urs.earthdata.nasa.gov/>) for GEDI access and, eventually, a Weights & Biases account (from <https://wandb.ai/>) for experiment tracking.

Copy the template and fill in your details:

```powershell
copy .env.example .env
```

For GEDI access specifically, persist your credentials to a `_netrc` file once:

```powershell
uv run python -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
```

After that, every script picks them up automatically.

### 4. Smoke test

Confirm everything is wired correctly:

```powershell
uv run pytest                                   # 4 tests should pass
uv run python scripts/01_query_gedi.py          # should report N granules in the dev AOI
```

---

## Hardware notes

This project is developed on a deliberately modest workstation: an NVIDIA GTX 1050 Ti with 4 GB VRAM, 16 GB system RAM, and Windows 11. That constraint shapes several design choices:

- Training is **patch-based regression** (small windows around each GEDI footprint), not full-scene dense regression. This keeps batch tensors small.
- Mixed-precision (AMP) is used throughout to halve the VRAM footprint of activations.
- Inference at AOI scale is **tiled and streamed** rather than fitting a whole tile in memory at once.
- Backbone models are kept small (≤3M parameters); the contribution is methodological (fusion strategy comparison), not architectural.

The pipeline is designed to be the same on a 4 GB consumer GPU as on a 24 GB workstation card. The result is reproducibility at low cost, and an explicit statement to reviewers that the result doesn't depend on hardware most labs don't have.

---

## Project layout

```
.
├── pyproject.toml             # Project metadata, dependencies, tool config
├── uv.lock                    # Locked dependency versions (committed)
├── .env.example               # Credential template
├── README.md
├── LICENSE                    # MIT
│
├── src/biomass/               # Importable package
│   ├── __init__.py
│   ├── config.py              # Constants: beams, variables, paths
│   ├── log_setup.py           # Logging setup
│   └── data/
│       ├── __init__.py
│       ├── aoi.py             # Named AOIs (dev, full)
│       ├── gedi.py            # GEDI L4A read and quality filter
│       └── gedi_pipeline.py   # End-to-end extraction pipeline
│
├── configs/                   # Hydra configs
│   ├── base.yaml
│   └── aoi/
│       ├── dev.yaml
│       └── full.yaml
│
├── scripts/                   # CLI entry points (numbered by phase)
│   ├── 01_query_gedi.py
│   ├── 02_prototype_granule_read.py
│   └── 03_extract_all_shots.py
│
├── tests/
│   ├── __init__.py
│   └── test_imports.py
│
├── docs/
│   └── decisions.md           # Methodology decision log
│
└── data/                      # gitignored (transient + outputs)
    ├── raw/
    ├── interim/
    └── processed/
```

---

## Reproducibility

- Dependency versions are pinned in `uv.lock` (committed).
- All scripts are Hydra-configurable; configs are versioned.
- Random seeds are set in every model config and logged with each W&B run.
- All non-obvious design choices are recorded in [`docs/decisions.md`](docs/decisions.md) with date and rationale — this also serves as the source material for the manuscript's methods section.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Acknowledgements

*To be added: data providers (NASA GEDI Science Team, ESA Copernicus Programme), advisors, and funding sources as the project develops.*

---

## Contact

**Chavosh Almassian**
- chavosh@outlook.com
- [LinkedIn](https://www.linkedin.com/in/chavosh-almassian-81a05216a/)
- [GitHub](https://github.com/Chavoshh)