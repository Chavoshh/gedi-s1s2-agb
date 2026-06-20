"""Regression metrics for biomass evaluation.

All metrics are computed in Mg/ha (raw label units). Predictions are clipped
at zero at metric computation time (negative biomass is unphysical) but the
underlying loss sees raw predictions.

Designed for streaming accumulation across batches: each batch updates running
sums; the final metric is computed once at epoch end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class StreamingRegressionMetrics:
    """Accumulate prediction / target sums across batches; compute metrics on demand."""

    n: int = 0
    sum_y: float = 0.0
    sum_yhat: float = 0.0
    sum_y2: float = 0.0
    sum_yhat2: float = 0.0
    sum_y_yhat: float = 0.0
    sum_abs_err: float = 0.0
    sum_sq_err: float = 0.0
    sum_signed_err: float = 0.0

    # For stratified metrics later: store everything if needed
    keep_all: bool = False
    all_y: list[np.ndarray] = field(default_factory=list)
    all_yhat: list[np.ndarray] = field(default_factory=list)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update running sums from a batch.

        Predictions are clipped at zero (no negative biomass).
        """
        y = target.detach().cpu().numpy().astype(np.float64)
        yhat = pred.detach().cpu().numpy().astype(np.float64)
        yhat = np.clip(yhat, 0.0, None)

        self.n += len(y)
        self.sum_y += y.sum()
        self.sum_yhat += yhat.sum()
        self.sum_y2 += (y * y).sum()
        self.sum_yhat2 += (yhat * yhat).sum()
        self.sum_y_yhat += (y * yhat).sum()
        err = yhat - y
        self.sum_abs_err += np.abs(err).sum()
        self.sum_sq_err += (err * err).sum()
        self.sum_signed_err += err.sum()

        if self.keep_all:
            self.all_y.append(y.astype(np.float32))
            self.all_yhat.append(yhat.astype(np.float32))

    def compute(self) -> dict[str, float]:
        """Compute the final metrics from accumulated sums."""
        if self.n == 0:
            return {"rmse": float("nan"), "mae": float("nan"),
                    "r2": float("nan"), "bias": float("nan"),
                    "rel_rmse_pct": float("nan")}

        n = self.n
        mean_y = self.sum_y / n
        var_y = self.sum_y2 / n - mean_y * mean_y           # population variance
        ss_tot = var_y * n                                  # sum of squared deviations from mean
        ss_res = self.sum_sq_err                            # sum of squared residuals

        rmse = float(np.sqrt(self.sum_sq_err / n))
        mae = float(self.sum_abs_err / n)
        bias = float(self.sum_signed_err / n)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rel_rmse_pct = float(rmse / mean_y * 100.0) if mean_y > 0 else float("nan")

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "bias": bias,
            "rel_rmse_pct": rel_rmse_pct,
        }

    def reset(self) -> None:
        self.n = 0
        self.sum_y = self.sum_yhat = 0.0
        self.sum_y2 = self.sum_yhat2 = self.sum_y_yhat = 0.0
        self.sum_abs_err = self.sum_sq_err = self.sum_signed_err = 0.0
        self.all_y.clear()
        self.all_yhat.clear()