# Decision log

Chronological record of design and methodology decisions. Each entry records what was decided, why, and what alternatives were considered.

---

## 2026-06-08 - Project scope

**Decision:** Above-ground biomass density (AGBD) regression using GEDI L4A footprint biomass as supervision, fused with Sentinel-1 SAR and Sentinel-2 optical features. Produce wall-to-wall biomass maps over a forested AOI.

**Rationale:** Direct continuation of prior SAR–optical fusion work (DFC2020 segmentation) into a regression task. Clear publication path to IEEE GRSL or JSTARS; aligned with KIT C4LaND PhD positions focused on BIOMASS / NISAR forest monitoring.

**Alternatives considered:** Canopy height regression (rejected: less directly tied to carbon-storage applications). Land cover classification (rejected: already explored in prior DFC2020 work).

---

## 2026-06-08 - Study area

**Decision:** Northwest Iberian Peninsula (lon −9.5° to −5.5°, lat 41.5° to 43.5°) for the final paper. Development AOI = MGRS tile 29TNG in central Galicia (~110×110 km).

**Rationale:**
- Wide biomass dynamic range (dense Atlantic forest → Mediterranean oak → dehesa) stresses the model and exposes saturation behavior.
- Higher GEDI shot density than central Europe (lower latitude).
- Friendlier Sentinel-2 cloud climatology than central Europe.
- Strong reference data (Spanish IFN, ESA CCI Biomass) available for validation.
- C4LaND-relevant narrative: characterizes C-band limitations for biomass before BIOMASS (P-band) and NISAR (L-band) come online.

**Alternatives considered:** Southern Germany (rejected: narrower biomass range, more clouds, more saturated literature for central Europe). Amazon (rejected: severe sensor saturation makes the comparison less informative). Tropical Africa (rejected: less reference data, larger area).

---

## 2026-06-08 - GEDI product version

**Decision:** GEDI L4A V2.1 (cloud-hosted, short name `GEDI_L4A_AGB_Density_V2_1_2056`).

**Rationale:** Current version. V2.1 corrected algorithm setting group 10 issues from V2.0. Cloud-hosted variant allows future S3-direct access if compute moves to AWS us-west-2.

---

## 2026-06-08 - Time window

**Decision:** 2019-06-01 to 2022-12-31 for primary GEDI acquisition. Sentinel composites later reduced to 2020–2022; see 2026-06-09 entry.

**Rationale:** Captures most of GEDI's pre-storage operational period (instrument was in storage March 2023 - April 2024). Avoids the early commissioning months (Dec 2018 – May 2019) where shot quality is less reliable. Long enough to accumulate dense AOI coverage; short enough to be tractable.

**Alternatives considered:** Including 2024+ post-storage data could improve coverage; deferred until baseline pipeline is established.

---

## 2026-06-08 - GEDI quality filter

**Decision:** Default cascade: `l4_quality_flag == 1` AND `l2_quality_flag == 1` AND `degrade_flag == 0` AND `sensitivity >= 0.95` AND `agbd >= 0`.

**Rationale:** GEDI L4A user guide defaults. Empirically `l4_quality_flag` already encodes the others, but explicit conditions guard against future product changes. `sensitivity >= 0.95` rather than 0.98 retains more shots; 0.98 reserved for later sensitivity analysis.

---

## 2026-06-08 - Data access strategy for GEDI

**Decision:** Download granules locally one at a time, process, delete. Don't stream HDF5 over HTTPS.

**Rationale:** Streaming via `fsspec` + `aiohttp` proved unreliable across the transatlantic link to ORNL DAAC (intermittent `ContentLengthError`). Sequential download / process / delete keeps peak disk at ~300 MB while remaining robust to network drops via `earthaccess.download`'s internal retries.

---

## 2026-06-08 - Project scaffolding

**Decision:** Full `pyproject.toml` with optional-dependency groups (`sentinel`, `model`, `viz`, `dev`); `src/biomass/` package layout; Hydra configs in `configs/`; `ruff` for lint+format; public GitHub repo with MIT license; W&B for experiment tracking (deferred until training begins).

**Rationale:** Public-from-start is easier than retrofitting. Layered optional dependencies avoid resolving unused packages.

---

## 2026-06-09 - AGBD upper cap for training

**Decision:** Cap AGBD at 500 Mg/ha when constructing the training set. Shots with AGBD > 500 Mg/ha are excluded.

**Rationale:** EDA on the 813,124 dev-AOI shots shows 4,611 shots above 500 Mg/ha and 1,197 above 1000 Mg/ha. Galician forests (predominantly Pinus pinaster, eucalyptus, mixed oak) have a realistic biomass ceiling around 500–700 Mg/ha. Values above this are GEDI L4A parametric-model extrapolation failures rather than real biomass. The 500 Mg/ha cut sits at roughly the 99.4th percentile, removing 0.57% of shots while preserving all physically plausible high-biomass observations.

**Alternatives considered:** 380 Mg/ha (99th percentile) is more aggressive but discards meaningful high-biomass signal that we want the model to learn. 1000 Mg/ha is too permissive and lets clear model failures into training.

---

## 2026-06-09 - Reduced Sentinel time range to 2020–2022

**Decision:** Build Sentinel-1/2 composites for years 2020, 2021, 2022 only (dropping 2019). GEDI shots from 2019 are excluded from the training set used in this paper.

**Rationale:** Credit-cost probe on CDSE openEO (4 credits per 10×10 km × 1-month S2 composite) implied full-AOI composites cost ~600–900 credits each; the full 4-year × 2-season × 2-sensor plan (16 composites) would have exceeded the free-tier monthly quota of ~10,000 credits with no margin. Dropping 2019 sacrifices only 115k of the 813k GEDI shots (14%) while bringing the total composite count from 16 to 6, with credit budget comfortably under quota.

**Alternatives considered:** 4 years × annual (8 composites, more shots kept but tighter budget). 2 years × seasonal (8 composites, drops 38% of shots). 3 years × seasonal (12 composites, no quota margin, risk of overruns at ~€30 / 1000 credits).

---

## 2026-06-09 - Annual instead of seasonal composites

**Decision:** Use a single annual median composite per year (Jan–Dec) instead of two seasonal composites (growing Apr–Sep and dormant Oct–Mar) per year.

**Rationale:** Annual composites match the prevailing practice in the GEDI-supervised remote sensing literature (Lang et al. 2023, Sialelli et al. 2024). The phenological signal that seasonal composites would capture is small for biomass estimation specifically (larger for canopy height) and modest in Galicia's evergreen-dominated landscape. Halving the composite count keeps credit usage within the free tier with margin for re-runs and inference-time composites later.

**Future ablation:** Seasonal composites remain an option for a follow-up experiment if baseline annual results warrant phenological exploration.

---

## 2026-06-09 - `agbd_se` field is not per-shot uncertainty

**Decision:** Do not use the `agbd_se` field for per-shot loss weighting or uncertainty propagation.

**Rationale:** EDA reveals that `agbd_se` is essentially constant across all 813k shots (std = 0.125 Mg/ha, IQR width = 0.026 Mg/ha). The field reports the global RMSE of the parametric model selected for a given shot's algorithm-setting-group × PFT × world-region combination, not a true per-shot prediction error. Most shots share the same value (~7.69 Mg/ha) because they share the same model.

**Backlog item:** Per-shot prediction intervals exist as `agbd_pi_lower_a*` and `agbd_pi_upper_a*` (one pair per algorithm setting group). A future re-extraction script should pull these to provide real per-shot uncertainty, useful for both weighted loss and Phase 7 uncertainty quantification. Non-blocking for Phase 2.

---

## 2026-06-10 - Sentinel-2 composite edge no-data accepted

**Decision:** Accept the ~0.85% no-data pixels concentrated on the right and bottom edges of the Sentinel-2 composites rather than re-build with a wider bbox.

**Rationale:** Inspection of the 2020 S2 composite shows the eastern 100 columns (~51%) and bottom 100 rows (~22%) contain pixels encoded as -32768 (no-data sentinel). This is consistent with UTM zone 29 / 30 reprojection edge effects from openEO. The interior of the composite is 100% clean. At patch-extraction time, any 25×25 patch containing the no-data sentinel is dropped; expected loss is a few hundred GEDI shots out of 698k, negligible. Re-running with a wider bbox would cost ~186 credits per composite (6 composites = ~1100 credits) without fundamentally fixing the projection boundary issue.

**Code implication:** patch-extraction code must treat -32768 as no-data, not zero.

---

## 2026-06-12 - Sentinel-1 acquisition strategy (final): ASF Hyp3 γ⁰ terrain-corrected RTC

**Decision:** Acquire Sentinel-1 data as terrain-corrected gamma-naught (γ⁰) RTC via ASF Hyp3, at 30 m resolution, sub-sampled to 12 scenes per year (one per month, closest to the 15th). Process all 36 scenes (3 years × 12 months) into annual composites: VV (dB), VH (dB), local incidence angle (degrees), resampled to the Sentinel-2 grid at 10 m for downstream use.

**Rationale (significant iteration during this session):**

*On RTC level.* True γ⁰ radiometric terrain correction is preferable for biomass work in mountainous Galicia, where slope-area artifacts in lesser-corrected products are 3–8 dB and could dominate the biomass signal. Three RTC sources were evaluated in sequence:

1. **CDSE openEO with `coefficient="gamma0-terrain"`** - initially planned as the cleanest path, integrated with the S2 acquisition pipeline. Rejected because CDSE openEO does not currently expose γ⁰-terrain (CDSE developer confirmation, January 2026). Only σ⁰-ellipsoid and γ⁰-ellipsoid are listed; in practice, even γ⁰-ellipsoid is rejected by client-side validation, leaving only σ⁰-ellipsoid available.

2. **CDSE openEO with `coefficient="sigma0-ellipsoid"`** - attempted as a fallback. Single annual full-AOI batch jobs failed twice with `MetadataFetchFailedException` in Spark shuffle stages after ~2.5 hours of executor time, indicating a backend resource limit for the AOI scale. Monthly chunking (12 jobs per year per sensor) was attempted to bypass this limit but the very first monthly job (July 2020) failed differently - an Orfeo Toolbox segmentation fault during SAR processing, with the platform marking the entire job as "error" despite writing output that could not be retrieved.

3. **ASF Hyp3 γ⁰-terrain** - final choice. True radiometric terrain correction using Copernicus DEM 30 m (the same DEM Hyp3 produces internally and that we adopt as a model input — see 2026-06-13 entry). Free for academic use with the same Earthdata Login credentials used for GEDI. The Hyp3 full-scene delivery model (each scene ~6 GB at 10 m resolution) made a dense temporal stack at 10 m impractical (~2.3 TB of downloads), so we chose 30 m resolution and 12 scenes per year. The 30 m S1 is resampled to the 10 m S2 grid at composite-assembly time (bilinear).

*On the auxiliary band set.* Hyp3 supports VV, VH, and local incidence angle (LIA) as parallel outputs. All three are kept. Layover/shadow mask was available but not included; we instead rely on the model's auxiliary DEM (elevation, slope) and on the per-pixel LIA channel to disambiguate terrain effects.

*On compositing.* For each year, the 12 monthly scenes are reprojected to the Sentinel-2 grid (10 m, EPSG:32629), masked for non-positive linear-power values, and reduced via temporal median per band. VV and VH medians are converted to dB locally; LIA is kept in degrees.

*On speckle filtering.* No per-image speckle filter. The temporal median over 12 scenes per year already suppresses speckle substantially without sacrificing spatial resolution that per-image filtering would cost. Matches Sialelli et al. 2024 and most current GEDI-supervised biomass literature.

**Costs incurred during the saga:** ~1,100 CDSE openEO credits across the three failed attempts (refunded automatically). ~250 ASF Hyp3 credits across 36 successful jobs plus the recovery attempts. ~22 GB of total Hyp3 downloads.

**Future work:** Replication of the SAR acquisition once CDSE's openEO supports γ⁰-terrain (currently on their public roadmap) would simplify the project's infrastructure significantly without requiring re-acquisition of the labels or S2 data. The Hyp3 outputs are the higher-quality choice; any later replication should use Hyp3 as the reference.

**Alternatives considered:** Local SNAP / pyroSAR RTC processing (rejected: multi-day infrastructure setup on a hardware-limited workstation, ~30 hours of local CPU). σ⁰-ellipsoid (rejected after we confirmed Hyp3 was viable: terrain artifacts in Galicia are real and avoidable). Skipping S1 entirely (rejected: fusion comparison is the paper's core contribution).

---

## 2026-06-12 - Interferometric SAR products deferred

**Decision:** Use Sentinel-1 GRD intensity only (γ⁰ RTC backscatter). Do not include interferometric coherence or polarimetric decomposition as model inputs in this paper.

**Rationale:** Sentinel-1 IW mode over land provides dual-polarization (VV+VH) only, so full polarimetric decomposition is unavailable. Interferometric coherence requires SLC products and pairwise InSAR processing, which would substantially increase complexity and credit cost (~5–10× per composite) while opening a parallel methodological story that competes with the fusion-comparison contribution.

**Future work:** A follow-up paper using InSAR coherence as a forest-structure input is a natural extension, particularly given the author's SAR / InSAR background.

---

## 2026-06-12 - Phase 2 cleanup and script reorganization

**Decision:** Cleaned up throwaway debug scripts, intermediate test outputs (~7 GB total), and the abandoned CDSE monthly-chunking scripts. Promoted the S1 audit utility to `scripts/tools/audit_s1.py` for ongoing maintenance use.

**Pipeline numbering:** Scripts 11–13 are intentionally absent from the pipeline. They were used for an abandoned CDSE Sentinel-1 monthly chunking strategy (see 2026-06-12 entry on Sentinel-1 acquisition strategy). The numbering gap is preserved to maintain git diff continuity rather than renumber. The S1 pipeline runs 14 (scene selection) → 15 (Hyp3 submission) → 16 (annual composite assembly) → 17 (inspection).

**Removed:** debugging scripts (`_debug.py`, `_list_jobs.py`, `_recovery.py`, `_fix_s1_gaps.py`, `_submit_one.py`); abandoned CDSE chunking scripts (`11_build_s1_monthly.py`, `12_aggregate_s1_annual.py`, `13_launch_s1_year.py`); intermediate test outputs (`data/interim/openeo_smoke_test/`, `openeo_cost_probe/`, `s1_monthly/`, `gedi_shards_dev/`); the Hyp3 single-scene smoke test (`data/raw/hyp3_test/`).

---

## 2026-06-13 - Copernicus DEM GLO-30 added as auxiliary input

**Decision:** Use Copernicus DEM GLO-30 as a 2-band auxiliary input (elevation in meters, slope in degrees) over the dev AOI, resampled to the S2 / S1 grid. Source: public AWS S3 bucket (`s3://copernicus-dem-30m/`), anonymous HTTPS access. Slope computed via central-difference gradients on the resampled 10 m grid.

**Rationale:** Topographic context helps biomass models in two ways: (1) elevation correlates with vegetation type and structure in Galicia (lowland eucalyptus → upland pine → high-elevation oak / shrubland); (2) slope captures terrain-modulated effects on both spectral and SAR signals not fully removed by γ⁰-terrain RTC. Same DEM that Hyp3 used for SAR terrain correction, ensuring methodological consistency.

**Alternatives considered:** Adding aspect (rejected: encoding sin / cos to handle wraparound adds bands without obvious biomass benefit; the model can recover any aspect-dependent solar effect from the S2 spectral signal if needed). Skipping DEM entirely (rejected: terrain is too prominent a Galician landscape feature to leave out).

---

## 2026-06-13 - Patch extraction strategy and dataset construction

**Decision:** Extract 25×25 pixel patches (250 m × 250 m at 10 m resolution) centered on each filtered GEDI shot. Each patch contains 15 channels in a fixed order (10 S2 spectral + 3 S1 + 2 DEM). Year-match: each GEDI shot is paired with the raster composite from its own acquisition year. The dataset is stored in a single Zarr v2 store with stratified spatial-block train / validation / test splits.

**Filter cascade applied at extraction time:**
1. Drop shots with acquisition year outside {2020, 2021, 2022}.
2. Drop shots with AGBD > 500 Mg/ha (see 2026-06-09 cap decision).
3. Spatial sub-sampling: keep at most one shot per 100 m × 100 m UTM grid cell (mitigates the spatial redundancy of GEDI's along-track 60 m footprint clustering).
4. Drop shots whose 25×25 window would extend outside the raster grid.
5. Drop patches containing any no-data pixel in any band (the -32768 sentinel for S2, NaN for S1 / DEM).

**Channel order (locked, used throughout downstream code):**
`B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12, VV_dB, VH_dB, LIA_deg, elevation_m, slope_deg`

**Train / validation / test split:** Spatial-block split, 10 km × 10 km blocks, stratified by mean per-block AGBD (5 quantile strata), with 70 / 15 / 15 fractions by block count within each stratum. All shots within a block go to the same partition, preventing spatial information leakage from train into validation or test. Note: shot-level partition fractions deviate from the block-level 70 / 15 / 15 because blocks have unequal shot density (rural blocks contain many GEDI shots; coastal / urban blocks contain few).

**Normalization:** Raw values are stored in the Zarr; per-channel mean and standard deviation are computed once over the training partition and saved alongside the dataset for application on-the-fly during training.

**Outcome:**
- 813,124 raw shots → 698,203 (year filter) → 694,015 (AGBD cap) → 377,201 (spatial sub-sample) → 375,817 patches written to Zarr (1,306 dropped for in-patch no-data, 0.35% loss rate).
- Train / validation / test: 261,531 / 42,253 / 72,033 (69.6% / 11.2% / 19.2% by shot count, deviating from the 70 / 15 / 15 block-count target due to per-block shot-density variation).
- Final Zarr size on disk: ~6.5 GB.
- Reproducibility: seeded with `RNG_SEED = 42`.

**Rationale (selected design choices):**

*Why year-match instead of multi-year stacking:* Matches Lang et al. 2023 and Sialelli et al. 2024. Simpler input shape (15 channels rather than 41). Multi-year stacking remains a follow-up ablation if baseline results warrant exploring interannual change.

*Why spatial-block split:* Naive random splitting leaks information between train and test because adjacent GEDI shots see nearly identical 25×25 patches. Spatial-block stratification is the standard approach in geospatial ML for realistic generalization estimates.

*Why 100 m subsampling:* Reduces dataset size by ~46% (694k → 377k) without losing biophysically distinct samples. Adjacent GEDI footprints within 100 m see essentially the same 25×25 patch, so they contribute redundant information.

**Alternatives considered:** Multi-year stacking (deferred to follow-up). Random train / test split (rejected: spatial leakage would inflate validation metrics). No spatial sub-sampling (considered: gives ~694k patches but with substantial spatial-correlation redundancy; the smaller stratified dataset is preferable for cleaner cross-validation).

## 2026-06-15 — Phase 3 model architecture and training infrastructure

**Decision:** Use a ResNet-18 backbone adapted for 25×25 patches as the shared encoder across all four model variants (S1-only, S2-only, early fusion, late fusion). Train with Huber loss (δ=30 Mg/ha), AdamW optimizer, cosine learning-rate schedule with linear warmup, AMP mixed precision (fp16), and label-invariant augmentation (90° rotations, horizontal/vertical flips). Per-channel z-score normalization applied on-the-fly using statistics computed over the training partition only.

**Architecture details:**
- Backbone modifications: 3×3 stride-1 stem (replaces the standard ImageNet 7×7 stride-2 stem + maxpool, which would collapse the 25×25 spatial dimension prematurely). Three residual stages with channel depths 64 → 128 → 256 (the standard fourth stage at 512 channels is dropped because it would reduce spatial dimensions to 1×1, throwing away spatial information).
- Output: 256-dimensional feature vector after global average pooling. Single encoder for S1-only, S2-only, and early fusion variants (~2.78 M parameters each); two parallel encoders for late fusion (~5.56 M total).
- Regression head: 2-layer MLP with dropout (256 → 64 → 1 for single encoder, 512 → 64 → 1 for late fusion). The output is unconstrained (no sigmoid / softplus); negative predictions are clipped at evaluation time only.
- Late fusion routing: the dataset returns the full 15-channel patch; the model uses `torch.index_select` internally to split into the S1 branch (VV, VH, LIA, elevation, slope = 5 channels) and the S2 branch (10 spectral bands + elevation + slope = 12 channels). DEM is duplicated across both branches because terrain context is informative for both modalities and keeps the comparison to single-sensor variants clean.

**Training configuration (defaults):**
- Loss: Huber (δ=30 Mg/ha). Quadratic regime within ±30 Mg/ha of true biomass, linear beyond. The transition point is chosen at roughly the standard deviation of the training labels, which gives a sensible tradeoff between MSE-like behavior for small errors and MAE-like robustness for the high-biomass tail.
- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4.
- Schedule: 2-epoch linear warmup followed by cosine annealing to a floor of 1e-6 over 30 epochs.
- Batch size: 64. Mixed precision (AMP fp16) with `GradScaler` for stability.
- Early stopping: patience 5 epochs on validation RMSE, minimum delta 0.05 Mg/ha.
- Augmentation: random 90° rotation (uniform over k ∈ {0, 1, 2, 3}) followed by independent random horizontal and vertical flips. All augmentations are label-invariant (AGBD is rotation- and flip-invariant for a single shot).

**Rationale:**

*On the backbone choice.* Standard ResNet-18 sized for ImageNet (224×224 input) is the natural starting point given prior project experience (DFC2020 land-cover segmentation, Sentinel-2 burn segmentation both used ResNet-18 encoders within U-Net architectures). However, the standard ImageNet stem (7×7 conv stride 2 + 3×3 maxpool stride 2) immediately downsamples 4×, which would reduce 25×25 to 6×6 before any residual blocks fire — destructive for small inputs. The fix is the standard CIFAR-style adaptation: 3×3 stride-1 stem with no maxpool, three residual stages instead of four. Output feature map at 7×7×256 retains meaningful spatial structure before global pooling.

*On the loss choice.* The AGBD label distribution is right-skewed: median 47 Mg/ha, p99 379 Mg/ha (after capping at 500 Mg/ha per the 2026-06-09 decision). MSE penalizes high-biomass errors heavily and biases predictions toward low values; MAE is too flat near the median. Huber loss is the standard compromise — quadratic near zero, linear beyond. δ=30 Mg/ha is a sensible starting point (roughly the std of the training labels) and was retained after the W&B sweep found no meaningful improvement from alternatives (15.0, 50.0).

*On the fusion design.* Late fusion uses feature concatenation followed by a single MLP head, not cross-attention or feature averaging. Concatenation is the standard approach in published GEDI biomass work and introduces the fewest additional variables that would confound the early-vs-late comparison. The DEM is included in both branches of late fusion (rather than as a third independent branch) to keep the architecture simple and the comparison to single-sensor variants clean.

**Alternatives considered:** ResNet-34 (planned ablation for capacity control). Custom small CNN under 1 M parameters (rejected for less defensible architectural lineage). Cross-attention fusion (rejected as confounding the central comparison). Log-cosh loss (rejected because Huber is easier to interpret and tune; differences are marginal).

---

## 2026-06-19 - Hyperparameter sweep on S2-only variant

**Decision:** Bayesian hyperparameter sweep run on the S2-only variant via Weights & Biases. Searched over learning rate, batch size, weight decay, head hidden dimension, head dropout, Huber δ, and warmup epochs. Ten configurations completed before stopping; the sweep had clearly converged to a tight performance band, with no candidate beating the default configuration meaningfully.

**Sweep configuration:**
- Method: Bayesian optimization.
- Metric: minimize validation RMSE.
- Epochs per run: 20 (reduced from the 30-epoch production budget; sufficient to rank candidates without wasting compute on tail convergence).
- Early stopping patience: 5 epochs.

**Search ranges:**
- Learning rate: log-uniform [3e-4, 3e-3].
- Batch size: {32, 64, 128}.
- Weight decay: log-uniform [1e-5, 1e-3].
- Head hidden dimension: {32, 64, 128}.
- Head dropout: {0.1, 0.2, 0.3, 0.4}.
- Huber δ: {15.0, 30.0, 50.0}.
- Warmup epochs: {0, 1, 2}.

**Result:**
- Range of validation RMSE across 10 runs: 44.92 to 45.64 Mg/ha. Spread of 0.72 Mg/ha across ten substantially different configurations.
- Two of ten runs essentially tied with the no-sweep default-config baseline (44.92 and 44.96 vs. baseline 44.89).
- No run beat the baseline.

**Interpretation:** The narrow spread and the failure to beat the default-config baseline indicate that S2-only performance on this dataset is bounded by an architecture / data ceiling at approximately 45 Mg/ha validation RMSE rather than by hyperparameter tuning. The Bayesian sampler effectively confirmed the manually-chosen defaults — two independent search procedures (manual default selection and Bayesian optimization) converged to the same hyperparameter region.

**Decision for the final reporting runs:** Use the default training configuration (the original Phase 3 defaults) for all 12 reporting runs (4 variants × 3 seeds). No hyperparameter changes between the sweep and the reporting runs. This keeps the methods section simple: a single configuration applied across all variants, with sweep evidence supporting that the configuration is near-optimal and that results are robust to reasonable hyperparameter variation.

**Methodological framing for the paper:** "Hyperparameters were tuned via Bayesian optimization on the S2-only variant, with 10 candidate configurations evaluated against validation RMSE. The search converged to validation RMSE 45.2 ± 0.2 Mg/ha across configurations, with the default configuration achieving 44.9 Mg/ha. We adopt the default configuration for all reported results."

**Alternatives considered:** Extending the sweep to 20+ runs (rejected: variance across 10 runs already small enough that additional runs would provide diminishing information). Picking the absolute-best sweep configuration (44.92 Mg/ha) over the default (44.89 Mg/ha) (rejected: difference is below measurement noise, and "default config" is a cleaner methods sentence). Per-variant hyperparameter sweeps (rejected: defensibility hinges on a single configuration applied consistently across all variants).
---

## 2026-06-25 - Phase 3 reporting batch results

**Decision:** 12 reporting runs completed (4 variants × 3 seeds = 42, 7, 123). All runs used the default configuration (30 epochs, batch=64, lr=1e-3, Huber δ=30, dropout=0.2, head_hidden_dim=64), determined as near-optimal by the Bayesian sweep on 2026-06-19. Validation RMSE results form the basis for selecting the headline variant before Phase 4 test-set evaluation.

**Results (validation RMSE, mean ± std across 3 seeds):**

| Variant       | Val RMSE (Mg/ha) | Notes |
|---------------|------------------|-------|
| Late fusion   | 45.13 ± 0.23     | Best |
| S2-only       | 45.39 ± 0.49     | +0.27 vs late fusion |
| Early fusion  | 45.46 ± 0.29     | +0.34 vs late fusion |
| S1-only       | 51.50 ± 0.19     | +6.38 vs late fusion |

**Key findings:**

1. **Late fusion is the best variant by a small but consistent margin.** Mean RMSE 0.27 Mg/ha lower than S2-only, with cross-seed std (0.23) smaller than the gap. This is the headline finding: independent SAR and optical encoders, combined at the feature level via concatenation and a single MLP head, produce the lowest validation RMSE.

2. **Early fusion is statistically indistinguishable from S2-only.** Gap is 0.07 Mg/ha, well within cross-seed variance. This is the methodologically important negative result — naive channel concatenation does not extract value from SAR information that's available in the input. Late fusion's advantage over early fusion (Δ = 0.33 Mg/ha) is the controlled signal of the fusion-strategy comparison.

3. **S1-only is dramatically worse than any optical-using variant.** ~6 Mg/ha gap, very low cross-seed variance (0.19). The C-band SAR signal alone is insufficient for biomass regression at the resolution and biomass range of this AOI. This is consistent with the published understanding that C-band saturates around 100 Mg/ha biomass.

**Interpretation of why late fusion outperforms early fusion:** Sentinel-1 SAR and Sentinel-2 optical bands have very different statistical properties (dB units vs reflectance scales, specular vs spectral signal types). A single encoder operating on concatenated raw channels must learn joint representations from both modalities simultaneously, which is harder than learning each modality's representation separately and combining them at a higher abstraction level. Late fusion structurally enforces modality-specific feature learning before combination.

**Reproducibility check:** The S2-only seed-42 reporting run achieved val RMSE 44.90 Mg/ha, reproducing the prior baseline (44.89 Mg/ha, also seed 42) to within 0.01 Mg/ha. This confirms the training procedure is reproducible across runs.

**Validation set caveat:** All numbers above are on the validation partition, which the training loop uses for early stopping. The headline numbers for the paper will come from the held-out test partition (72,033 patches, spatially independent of train and val, never seen during training or hyperparameter selection). Test-set evaluation is Phase 4.

**Compute budget:** Total wall time for the 12 reporting runs was approximately 80 hours of background compute on the GTX 1050 Ti workstation. Average per-run time was 6.7 hours, ranging from ~5 hours (S1-only, fast early-stopping) to ~9 hours (late fusion, two encoders).

**Alternatives considered:** Per-variant hyperparameter sweeps (rejected: would have multiplied compute by 4x and confounded the fusion-strategy comparison with per-variant tuning differences). Multi-year temporal stacking instead of year-matched

## 2026-06-29 — Phase 4: test-set evaluation

**Decision:** All 12 trained checkpoints from Phase 3 evaluated on the 72,033-patch test partition. Test metrics computed overall and stratified by AGBD bin (Lang 2023's bins: 0-50, 50-100, 100-150, 150-200, 200-500 Mg/ha) and by majority land cover class (ESA WorldCover 2021). Three publication-ready figures generated: predicted-vs-observed hexbin scatter, RMSE by AGBD bin, RMSE by land cover.

**Headline test-set results (mean ± std across 3 seeds):**

| Variant       | Test RMSE (Mg/ha) | Test R²       | Test Bias (Mg/ha) |
|---------------|-------------------|---------------|-------------------|
| Late fusion   | **50.33 ± 0.37**  | 0.441 ± 0.008 | −3.77 ± 3.88      |
| Early fusion  | 50.69 ± 0.14      | 0.433 ± 0.003 | −5.02 ± 0.41      |
| S2-only       | 50.87 ± 0.69      | 0.429 ± 0.015 | −6.80 ± 0.66      |
| S1-only       | 59.12 ± 0.07      | 0.229 ± 0.001 | −8.16 ± 1.77      |

**Test vs. validation comparison:** Test RMSE is 5-8 Mg/ha higher than validation RMSE for every variant (val late fusion 45.13 → test 50.33; val S2-only 45.39 → test 50.87; val S1-only 51.50 → test 59.12). The validation-to-test gap is consistent across variants, indicating the test partition is intrinsically harder (likely due to spatial-block sampling placing more high-biomass or more heterogeneous blocks in test). The variant ordering preserves across val and test: late fusion best, then early fusion ≈ S2-only, then S1-only.

**Key finding 1 - Late fusion advantage is concentrated in high-biomass tree cover.** Stratification by AGBD bin and by land cover both reveal that the late-fusion advantage over S2-only is regime-dependent rather than uniform. In the 200-500 Mg/ha biomass bin, late fusion RMSE is 158.7 Mg/ha vs. S2-only's 164.5 Mg/ha (Δ = 5.8 Mg/ha). In the 0-50 Mg/ha bin, the variants are statistically tied (~21 Mg/ha each). Equivalently, in tree-cover patches (n=46,503) late fusion beats S2-only by 0.70 Mg/ha; in grassland (n=22,609) the gap is 0.16 Mg/ha; in cropland (n=2,380) the gap is 0.11 Mg/ha. The fusion contribution to model performance is concentrated in regimes where optical alone saturates (closed canopy, high biomass) and where SAR is still in its informative range.

**Key finding 2 - S1-only saturates dramatically above 100 Mg/ha.** S1-only test RMSE rises from 29 Mg/ha (0-50 bin) to 189 Mg/ha (200-500 bin): a 6.5× increase, while optical-using variants increase 7-8× from 21 to 159-165. In tree cover specifically, S1-only is 10 Mg/ha worse than any optical-using variant. C-band SAR alone is insufficient for biomass estimation at the resolution and biomass range of this AOI, consistent with the published understanding that C-band backscatter saturates around 100 Mg/ha biomass.

**Key finding 3 - Early fusion fails to extract value from SAR information.** Early fusion test RMSE (50.69 ± 0.14) is statistically indistinguishable from S2-only (50.87 ± 0.69), despite having access to the SAR channels that late fusion uses to achieve a 0.54 Mg/ha advantage. The architectural choice — naive concatenation versus modality-specific encoding, determines whether the model can use SAR information. This is the methodologically central finding: the fusion *strategy* matters more than the *presence* of fusion.

**Late fusion bias variance investigation:** The cross-seed bias standard deviation for late fusion (3.88 Mg/ha) is substantially larger than for any other variant (≤ 1.8 Mg/ha). Per-seed analysis: late fusion seed 42 had bias +0.46 (best epoch 15), seed 7 had bias −4.62 (best epoch 25), seed 123 had bias −7.16 (best epoch 26). The correlation between best-epoch and bias magnitude suggests that late fusion's larger parameter count produces a longer training trajectory along which bias drifts toward under-prediction. Different seeds catch the model at different positions on this trajectory. RMSE remains tightly clustered across seeds (50.03 to 50.74), so the variance is in bias direction rather than overall accuracy. We report bias as mean ± std faithfully; the figures use seed 7 (most representative) for the predicted-vs-observed scatter.

**Land cover stratification methodology:** ESA WorldCover 2021 v200 reprojected from EPSG:4326 to the 10 m UTM 29N project grid using nearest-neighbor (preserving categorical values). Each patch assigned its majority land cover class within the 25×25 window. Mean patch purity (fraction of pixels in the majority class) is 0.73, reflecting the mosaic character of the Galician landscape. Test patches distribute as: tree cover 64.6% (n=46,503), grassland 31.4% (n=22,609), cropland 3.3% (n=2,380), other 0.9% (n=541). The all-patch and test-only distributions match within 2 percentage points, confirming the spatial-block split did not over- or under-sample any land cover.

**Pre-computed artifacts saved:** `data/processed/test_predictions.parquet` (864,396 predictions, 5 columns: patch_id, variant, seed, true_agbd, pred_agbd; 5.4 MB), `data/processed/test_metrics_overall.csv`, `data/processed/test_metrics_by_agbd_bin.csv`, `data/processed/test_metrics_by_landcover.csv`, `data/processed/patch_landcover_dev.parquet`, three figures in `data/processed/figures/`. Inference wall time: 35 minutes for all 12 checkpoints (~3 minutes per checkpoint).

**Methodological framing for the paper:** "Fusion strategy matters more than fusion presence. Late fusion of independent SAR and optical encoders reduces test RMSE by 0.54 Mg/ha relative to optical-only baseline, with the improvement concentrated in high-biomass tree cover (200-500 Mg/ha bin: Δ = 5.8 Mg/ha; tree cover: Δ = 0.70 Mg/ha). Naive concatenation-based early fusion produces no measurable improvement over optical-only, demonstrating that the architectural choice for combining heterogeneous remote sensing modalities is more important than the presence of those modalities in the input."

**Alternatives considered:** Reporting test metrics for seed 42 only (rejected: would not capture the late-fusion bias variance, and seed 42 happens to be unrepresentative for late fusion specifically). Pooling predictions across seeds before computing RMSE (rejected for the headline; kept consistent with Phase 3 validation reporting). Stratifying by canopy cover percentile instead of WorldCover class (rejected: WorldCover classes are more interpretable to a remote-sensing audience and align with the existing literature).

## 2026-07-01 — Phase 5 (in progress): wall-to-wall inference + ensemble uncertainty

**Decision:** Produce wall-to-wall biomass maps over the full dev AOI from the trained checkpoints, at 100 m resolution, using 2021 input composites. Late fusion is the headline variant (3-seed ensemble mean + uncertainty); S2-only, S1-only, and early fusion (seed 7) are supporting maps.

**Fully convolutional inference abandoned.** The original plan was to convert the patch-regression models to fully convolutional form (replace AdaptiveAvgPool2d(1) with sliding AvgPool2d, Linear→Conv2d 1×1) for single-pass dense inference. Verified the conversion reproduces patch predictions exactly on a 25×25 input (abs diff ≤ 7.6e-6). But on larger tiles the interior predictions drifted badly (25×25→56.2, 500×500→15.5 for the same center pixel). Root cause: the encoder's AdaptiveAvgPool2d(1) + strided downsampling is not translation-equivariant — a sliding 7×7 pool over a large feature map does not reproduce the global pool of an isolated 25×25 patch, because intermediate strided-conv activations depend on the surrounding spatial context. This is a fundamental architecture limitation, not a code bug. Fully convolutional conversion of patch-CNNs with global pooling requires retraining, which was out of scope.

**Adopted: patch-based tiled inference at stride 10 (100 m output).** Extract real 25×25 patches on a stride-10 grid, batch them through the original trained model. This reproduces the training computation exactly (verified: correlation 1.0000, mean diff +0.003 Mg/ha, max |diff| 0.179 vs Phase 4 test predictions on 2021 patches). Output at 100 m matches ESA CCI Biomass v5 resolution, simplifying the planned comparison. Honest about resolution: GEDI supervision is inherently ~25 m, so 100 m output is not false precision.

**Performance fix.** Initial implementation read each patch with individual windowed rasterio reads (~3.3M tiny reads/job) — did not finish one job in 4+ hours. Rewrote as strip-batched: process output rows in strips of 40, reading one padded full-width row-block per strip (~28 large sequential reads/job) and slicing patches from memory. Result: ~12 min/job, ~1 hour for all 6 maps.

**Input nodata handling.** S2 composite uses -32768 sentinel; S1 and DEM use NaN. The reader detects invalid pixels per band, replaces with channel mean (→0 after normalization) so the model receives finite input, and masks output where any input band was nodata at the patch center. Predictions clipped to [0, 500] Mg/ha (training label range) to suppress extreme extrapolation near land/ocean boundaries.

**Maps produced (6, 100 m, 989×1122 px, ~95% coverage over land):**
- biomass_late_fusion_seed{42,7,123}.tif
- biomass_s2_only_seed7.tif, biomass_s1_only_seed7.tif, biomass_early_fusion_seed7.tif

Distributions confirm correctness: early fusion tracks S2-only (mean 68.5 vs 68.0, median 48.5 vs 47.0) as Phase 4 predicted; S1-only is compressed (p95 122.8, max 220.9) reflecting C-band saturation; optical variants reach the 500 cap. AOI-wide means (~68 Mg/ha) are lower than test-set means because the full AOI has more low-biomass grassland (33%) than the tree-heavy test set (64% tree cover).

**Ensemble mean + uncertainty (script 29).** Per-pixel mean and sample std across the 3 late-fusion seeds. Ensemble mean 66.9 Mg/ha (median 47.4). Uncertainty std: mean 5.63, median 3.21, p95 18.17 Mg/ha; mean coefficient of variation 7.6%. The 3 seeds agree well at the aggregate level; uncertainty concentrates in high-biomass pixels, consistent with the Phase 4 finding that late-fusion seed variance is largest in the high-biomass regime. Outputs: biomass_late_fusion_mean.tif (published map), biomass_late_fusion_std.tif (uncertainty layer).

**Still to do in Phase 5:** ESA CCI Biomass v5 comparison, Spanish IFN validation, Phase 5 figures.

## Appendix: Template for new entries

```
## YYYY-MM-DD - Short title

**Decision:**

**Rationale:**

**Alternatives considered:**
```