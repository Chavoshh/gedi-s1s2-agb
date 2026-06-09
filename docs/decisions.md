# Decision log

Chronological record of design and methodology decisions. Each entry: date, decision, rationale, alternatives considered.

---

## 2026-06-08 - 
 Project scope

**Decision:** Above-ground biomass regression using GEDI L4A footprint biomass as supervision, fused with Sentinel-1 SAR and Sentinel-2 optical features. Produce wall-to-wall biomass maps over a forested AOI.

**Rationale:** Direct continuation of prior SAR-optical fusion work (DFC2020 segmentation) into a regression task with clear publication path (GRSL / JSTARS).

---

## 2026-06-08 - Study area

**Decision:** Northwest Iberian Peninsula (lon −9.5° to −5.5°, lat 41.5° to 43.5°) for the paper. Development AOI = MGRS tile 29TNG in central Galicia (~110×110 km).

**Rationale:**
- Wide biomass dynamic range (dense Atlantic forest → Mediterranean oak → dehesa) stresses model and exposes saturation behavior.
- Higher GEDI shot density than central Europe (lower latitude).
- Friendlier S2 cloud climatology than central Europe.
- Strong reference data (Spanish IFN, ESA CCI Biomass) available for validation.
- C4LaND-relevant narrative: characterizes C-band limitations for biomass before BIOMASS (P-band) and NISAR (L-band) come online.

**Alternatives considered:** Southern Germany (rejected: narrower biomass range, more clouds, more saturated literature for central Europe). Amazon (rejected: severe sensor saturation makes the comparison less informative). Tropical Africa (rejected: less reference data, larger area).

---

## 2026-06-08 - GEDI product version

**Decision:** GEDI L4A V2.1 (cloud-hosted, short name `GEDI_L4A_AGB_Density_V2_1_2056`).

**Rationale:** Current version. V2.1 corrected algorithm setting group 10 issues from V2.0. Cloud-hosted variant allows future S3-direct access if compute moves to AWS us-west-2.

---

## 2026-06-08 - Time window

**Decision:** 2019-06-01 to 2022-12-31 for primary analysis.

**Rationale:** Captures most of GEDI's pre-storage operational period (instrument was in storage March 2023 – April 2024). Avoids the early commissioning months (Dec 2018 – May 2019) where shot quality is less reliable. Long enough to accumulate dense AOI coverage; short enough to be tractable.

**Alternatives:** Including 2024+ post-storage data could improve coverage; deferred until baseline pipeline is established.

---

## 2026-06-08 - GEDI quality filter

**Decision:** Default cascade: `l4_quality_flag == 1` AND `l2_quality_flag == 1` AND `degrade_flag == 0` AND `sensitivity >= 0.95` AND `agbd >= 0`.

**Rationale:** GEDI L4A user guide defaults. Empirically `l4_quality_flag` already encodes the others, but explicit conditions guard against future product changes. `sensitivity >= 0.95` rather than 0.98 retains more shots; 0.98 reserved for later sensitivity analysis.

---

## 2026-06-08 - Data access strategy

**Decision:** Download granules locally one at a time, process, delete. Don't stream HDF5 over HTTPS.

**Rationale:** Streaming via `fsspec` + `aiohttp` proved unreliable across the transatlantic link to ORNL DAAC (intermittent `ContentLengthError`). Sequential download/process/delete keeps peak disk at ~300 MB while remaining robust to network drops via `earthaccess.download`'s internal retries.

---

## 2026-06-08 - Project scaffolding

**Decision:** Full `pyproject.toml` with optional-dependency groups (`sentinel`, `model`, `viz`, `dev`); `src/biomass/` package layout; Hydra configs in `configs/`; `ruff` for lint+format; public GitHub repo with MIT license; W&B for experiment tracking (deferred until training begins).

**Rationale:** Public-from-start is easier than retrofitting. Layered optional deps avoid resolving unused packages.

## 2026-06-09 - AGBD upper cap for training

**Decision:** Cap AGBD at 500 Mg/ha when constructing the training set. Shots with AGBD > 500 Mg/ha are excluded.

**Rationale:** EDA on the 813,124 dev-AOI shots shows 4,611 shots above 500 Mg/ha and 1,197 above 1000 Mg/ha. Galician forests (predominantly Pinus pinaster, eucalyptus, mixed oak) have a realistic biomass ceiling around 500–700 Mg/ha. Values above this are GEDI L4A parametric model extrapolation failures rather than real biomass. The 500 Mg/ha cut sits at roughly the 99.4th percentile, removing 0.57% of shots while preserving all physically plausible high-biomass observations.

**Alternatives considered:** 380 Mg/ha (99th percentile) is more aggressive but discards meaningful high-biomass signal that we want the model to learn. 1000 Mg/ha is too permissive and lets clear model failures into training.

## 2026-06-09 - Reduced Sentinel time range to 2020-2022

**Decision:** Build Sentinel-1/2 composites for years 2020, 2021, 2022 only (dropping 2019). GEDI shots from 2019 will be excluded from the training set used in this paper.

**Rationale:** Credit-cost probe on openEO (4 credits per 10×10 km × 1-month S2 composite) implied full-AOI composites cost ~600–900 credits each; the full 4-year × 2-season × 2-sensor plan (16 composites) would have exceeded the free-tier monthly quota of ~10,000 credits with no margin. Dropping 2019 sacrifices only 115k of the 813k GEDI shots (14%) while bringing the total composite count from 16 to 6, with credit budget comfortably under quota.

**Alternatives considered:** 4 years × annual (8 composites, more shots kept but tighter budget). 2 years × seasonal (also 8 composites, drops 38% of shots). 3 years × seasonal (12 composites, no quota margin and risk of overruns at ~EUR 30/1000 credits).

---

## 2026-06-09 - Annual instead of seasonal composites

**Decision:** Use a single annual median composite per year (Jan-Dec) instead of two seasonal composites (growing Apr-Sep and dormant Oct-Mar) per year.

**Rationale:** Annual composites match the prevailing practice in the GEDI-supervised remote sensing literature (Lang et al. 2023, Sialelli et al. 2024). The phenological signal that seasonal composites would capture is small for biomass estimation specifically (larger for canopy height) and modest in Galicia's evergreen-dominated landscape. Halving the composite count keeps credit usage within free tier with margin for re-runs and inference-time composites later.

**Future ablation:** Seasonal composites remain an option for a follow-up experiment if baseline annual results warrant phenological exploration.
---

## 2026-06-09 - `agbd_se` field is not per-shot

**Decision:** Do not use the `agbd_se` field for per-shot loss weighting or uncertainty propagation.

**Rationale:** EDA reveals that `agbd_se` is essentially constant across all 813k shots (std = 0.125 Mg/ha, IQR width = 0.026 Mg/ha). The field reports the global RMSE of the parametric model that was selected for a given shot's algorithm setting group × PFT × world region, not a true per-shot prediction error. Most shots share the same value (~7.69 Mg/ha) because they share the same model.

**Backlog item:** Per-shot prediction intervals exist as `agbd_pi_lower_a*` and `agbd_pi_upper_a*` (one pair per algorithm setting group). A future re-extraction script should pull these to provide real per-shot uncertainty, useful for both weighted loss and Phase 7 uncertainty quantification. Non-blocking for Phase 2.
---

## Template

When adding a new decision, copy this and fill in:

\`\`\`
## YYYY-MM-DD — Short title

**Decision:**

**Rationale:**

**Alternatives considered:**
\`\`\`