# Decision log

Chronological record of design and methodology decisions. Each entry: date, decision, rationale, alternatives considered.

---

## 2026-06-08 — Project scope

**Decision:** Above-ground biomass regression using GEDI L4A footprint biomass as supervision, fused with Sentinel-1 SAR and Sentinel-2 optical features. Produce wall-to-wall biomass maps over a forested AOI.

**Rationale:** Direct continuation of prior SAR-optical fusion work (DFC2020 segmentation) into a regression task with clear publication path (GRSL / JSTARS).

---

## 2026-06-08 — Study area

**Decision:** Northwest Iberian Peninsula (lon −9.5° to −5.5°, lat 41.5° to 43.5°) for the paper. Development AOI = MGRS tile 29TNG in central Galicia (~110×110 km).

**Rationale:**
- Wide biomass dynamic range (dense Atlantic forest → Mediterranean oak → dehesa) stresses model and exposes saturation behavior.
- Higher GEDI shot density than central Europe (lower latitude).
- Friendlier S2 cloud climatology than central Europe.
- Strong reference data (Spanish IFN, ESA CCI Biomass) available for validation.
- C4LaND-relevant narrative: characterizes C-band limitations for biomass before BIOMASS (P-band) and NISAR (L-band) come online.

**Alternatives considered:** Southern Germany (rejected: narrower biomass range, more clouds, more saturated literature for central Europe). Amazon (rejected: severe sensor saturation makes the comparison less informative). Tropical Africa (rejected: less reference data, larger area).

---

## 2026-06-08 — GEDI product version

**Decision:** GEDI L4A V2.1 (cloud-hosted, short name `GEDI_L4A_AGB_Density_V2_1_2056`).

**Rationale:** Current version. V2.1 corrected algorithm setting group 10 issues from V2.0. Cloud-hosted variant allows future S3-direct access if compute moves to AWS us-west-2.

---

## 2026-06-08 — Time window

**Decision:** 2019-06-01 to 2022-12-31 for primary analysis.

**Rationale:** Captures most of GEDI's pre-storage operational period (instrument was in storage March 2023 – April 2024). Avoids the early commissioning months (Dec 2018 – May 2019) where shot quality is less reliable. Long enough to accumulate dense AOI coverage; short enough to be tractable.

**Alternatives:** Including 2024+ post-storage data could improve coverage; deferred until baseline pipeline is established.

---

## 2026-06-08 — GEDI quality filter

**Decision:** Default cascade: `l4_quality_flag == 1` AND `l2_quality_flag == 1` AND `degrade_flag == 0` AND `sensitivity >= 0.95` AND `agbd >= 0`.

**Rationale:** GEDI L4A user guide defaults. Empirically `l4_quality_flag` already encodes the others, but explicit conditions guard against future product changes. `sensitivity >= 0.95` rather than 0.98 retains more shots; 0.98 reserved for later sensitivity analysis.

---

## 2026-06-08 — Data access strategy

**Decision:** Download granules locally one at a time, process, delete. Don't stream HDF5 over HTTPS.

**Rationale:** Streaming via `fsspec` + `aiohttp` proved unreliable across the transatlantic link to ORNL DAAC (intermittent `ContentLengthError`). Sequential download/process/delete keeps peak disk at ~300 MB while remaining robust to network drops via `earthaccess.download`'s internal retries.

---

## 2026-06-08 — Project scaffolding

**Decision:** Full `pyproject.toml` with optional-dependency groups (`sentinel`, `model`, `viz`, `dev`); `src/biomass/` package layout; Hydra configs in `configs/`; `ruff` for lint+format; public GitHub repo with MIT license; W&B for experiment tracking (deferred until training begins).

**Rationale:** Public-from-start is easier than retrofitting. Layered optional deps avoid resolving unused packages.

---

## Template

When adding a new decision, copy this and fill in:

\`\`\`
## YYYY-MM-DD — Short title

**Decision:**

**Rationale:**

**Alternatives considered:**
\`\`\`