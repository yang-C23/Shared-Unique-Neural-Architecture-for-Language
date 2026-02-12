#!/usr/bin/env python3
"""
Heatmap visualization for core linguistic region masks.

Given a binary mask stored as a PyTorch tensor, this script applies the
“vicinity density” smoothing described in the paper “Unveiling Linguistic
Regions in Large Language Models” and renders the result as a heatmap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F


def _load_mask_tensor(file_path: Path) -> torch.Tensor:
    """Load a binary mask tensor and ensure it lives on CPU as float32."""
    tensor = torch.load(file_path, map_location=torch.device("cpu"))
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2D tensor, but got shape {tuple(tensor.shape)} for {file_path}"
        )
    # Convert to float to support convolution math; tolerate bool tensors.
    return tensor.to(dtype=torch.float32, copy=False)


def _compute_vicinity_density(mask_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply a 3x3 convolutional smoothing kernel and normalize by 9 so that
    each entry reflects the proportion of “Top” parameters in its vicinity.
    """
    kernel = torch.ones((1, 1, 3, 3), dtype=mask_tensor.dtype, device=mask_tensor.device)
    smoothed = F.conv2d(
        mask_tensor.unsqueeze(0).unsqueeze(0),  # NCHW
        kernel,
        padding=1,
        stride=1,
    )
    return (smoothed / 9.0).squeeze(0).squeeze(0).clamp_(0.0, 1.0)


def _major_ticks(length: int, num_ticks: int = 6) -> list[int]:
    """Return evenly spaced tick positions covering the axis range."""
    if length <= num_ticks:
        return list(range(length))
    ticks = np.linspace(0, length - 1, num=num_ticks, dtype=int)
    return sorted(set(ticks.tolist()))


def plot_linguistic_heatmap(file_path: str | Path, output_path: str | Path) -> None:
    """
    Visualize a binary linguistic-region mask using the 3x3 vicinity density metric.

    Args:
        file_path: Path to the `.pt` tensor file containing a 2D binary mask.
        output_path: Destination path for the rendered heatmap image (PNG, PDF, etc.).
    """
    file_path = Path(file_path)
    output_path = Path(output_path)

    mask_tensor = _load_mask_tensor(file_path)
    density_tensor = _compute_vicinity_density(mask_tensor)

    density_np = density_tensor.numpy()
    height, width = density_np.shape

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    heatmap = sns.heatmap(
        density_np,
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Vicinity Density"},
        ax=ax,
        square=False,
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Vicinity Density", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    heatmap.invert_yaxis()  # Row index 0 appears at the top, matching tensor layout.
    ax.set_xlabel("Column", fontsize=16)
    ax.set_ylabel("Row", fontsize=16)
    ax.set_title(
        f"Vicinity Density Heatmap\n{file_path.name}",
        fontsize=20,
        fontweight="bold",
        pad=25,
    )

    x_ticks = _major_ticks(width)
    y_ticks = _major_ticks(height)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks], fontsize=14, rotation=0)
    ax.set_yticklabels([str(t) for t in y_ticks], fontsize=14, rotation=0)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render vicinity density heatmaps for core linguistic region masks."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the binary mask `.pt` file (e.g., model.layers.X.weight.pt).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Destination image path (e.g., outputs/layer_X_heatmap.png).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_linguistic_heatmap(args.file_path, args.output_path)

