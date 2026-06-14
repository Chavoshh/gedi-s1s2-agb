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

---

## Appendix: Template for new entries

```
## YYYY-MM-DD — Short title

**Decision:**

**Rationale:**

**Alternatives considered:**
```