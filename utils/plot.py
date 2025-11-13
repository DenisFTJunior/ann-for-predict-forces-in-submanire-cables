"""Plotting helpers for regression models.

Enhancements:
- plot_predictions now supports multi-output (2D) targets by generating one figure per target.
- Optional y_names parameter to label each target's plots.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np


def _plot_predictions_single(y_true_1d, y_pred_1d, title: str, save_path: Optional[str] = None):
    import matplotlib.pyplot as plt  # imported here to keep import time light

    y_true_1d = np.asarray(y_true_1d).ravel()
    y_pred_1d = np.asarray(y_pred_1d).ravel()
    idx = np.arange(len(y_true_1d))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Top Graph: actual vs predicted over sample index
    axes[0].plot(idx, y_true_1d, label="Actual", marker="o", ms=3, lw=1)
    axes[0].plot(idx, y_pred_1d, label="Predicted", marker="x", ms=3, lw=1, linestyle="--")
    axes[0].set_ylabel("Value")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.2)

    # Bottom Graph: residuals (pred - actual)
    residuals = y_pred_1d - y_true_1d
    colors = np.where(residuals >= 0, "tab:blue", "tab:red")
    axes[1].axhline(0.0, color="k", lw=1)
    axes[1].bar(idx, residuals, color=colors, width=0.8)
    axes[1].set_ylabel("Pred - Actual")
    axes[1].set_xlabel("Sample Index")
    axes[1].grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig, axes


def plot_predictions(
    y_true,
    y_pred,
    title: str = "Predictions vs Actuals",
    save_path: Optional[str] = None,
    y_names: Optional[Sequence[str]] = None,
):
    """Plot predictions vs actuals and residuals.

    - If y_true/y_pred are 1D, produces one figure.
    - If they are 2D (multi-output), produces one figure per target (column).

    When save_path is provided for multi-output, files are saved with suffixes
    like "_y0.png", "_y1.png" (or using y_names if provided) next to save_path.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Multi-output case
    if y_true_arr.ndim == 2 and y_true_arr.shape[1] > 1:
        n_out = y_true_arr.shape[1]
        figs_axes = []
        root, ext = os.path.splitext(save_path) if save_path else (None, None)
        for i in range(n_out):
            name = None
            if y_names and i < len(y_names):
                name = y_names[i]
            suffix = f"_{name}" if name else f"_y{i}"
            title_i = f"{title}{' - ' + name if name else suffix}"
            save_i = f"{root}{suffix}{ext}" if save_path else None
            fig_ax = _plot_predictions_single(y_true_arr[:, i], y_pred_arr[:, i], title_i, save_i)
            figs_axes.append(fig_ax)
        # Return the last figure/axes for backward-compatibility
        return figs_axes[-1]

    # Single-output case
    return _plot_predictions_single(y_true_arr, y_pred_arr, title, save_path)
