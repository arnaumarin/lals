"""Plots.

    token_heatmaps(images, maps, titles)    token-level LALS drawn over the images (as in Figure 3)
    layer_sweep_figure(data_dir)            LALS across layers, per model (Figure 10)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def token_heatmaps(images: list, maps: list[list[np.ndarray]], titles: list[str],
                   row_labels: list[str]) -> plt.Figure:
    """Top row: the images. One row of heatmaps per entry of ``maps`` (e.g. one per layer), each a
    list of token grids (``LALSScorer.token_map``) aligned with ``images``. Red = female-leaning,
    blue = male-leaning; one colour scale for the whole figure."""
    values = np.concatenate([g[~np.isnan(g)] for row in maps for g in row])
    vmax = max(np.percentile(np.abs(values), 98), 0.05)
    n_rows, n_cols = len(maps) + 1, len(images)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)
    for col, (image, title) in enumerate(zip(images, titles)):
        axes[0, col].imshow(image)
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")
        for row, grids in enumerate(maps, start=1):
            ax = axes[row, col]
            ax.imshow(image, alpha=0.25)
            im = ax.imshow(np.nan_to_num(grids[col]), cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.75,
                           interpolation="bilinear", extent=(0, image.width, image.height, 0))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    for row, label in enumerate(row_labels, start=1):
        axes[row, 0].set_ylabel(label, fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="LALS (+ female / \u2212 male)")
    return fig


PANELS = [("qwen2vl", "Qwen2-VL"), ("qwen25vl", "Qwen2.5-VL"),
          ("llava", "LLaVA"), ("internvl", "InternVL")]

# Published per-model normalisers: the largest |occupation-mean LALS| over all layers and
# all occupations of the sweep files. The files also hold two occupations that are not
# plotted, and they set two of the maxima: "photographer" (Qwen2-VL, layer 24) and
# "paralegal" (office worker; LLaVA, layer 27).
NORMALISER = {"qwen2vl": 0.07661933109562154, "qwen25vl": 0.08120709724364554,
              "llava": 0.09384506688870216, "internvl": 0.13198324110639123}

# Colour and legend order: occupations ranked male -> female by their mean LALS pooled over
# all images, layers and models of the published data.
RANKED = ["firefighter", "delivery_driver", "construction", "flight_attendant", "cleaning",
          "scientist", "chef", "waiter", "babysitter", "hairdresser", "librarian",
          "preschool_teacher", "florist", "nurse", "makeup_artist"]

# Data key -> legend label, for the 15 occupations shown in the paper.
OCC_LABELS = {
    "construction": "Construction", "firefighter": "Firefighter",
    "flight_attendant": "Pilot", "delivery_driver": "Delivery Driver", "chef": "Chef",
    "cleaning": "Maids/Cleaning", "waiter": "Waiter", "scientist": "Scientist",
    "babysitter": "Babysitter", "hairdresser": "Hairdresser", "librarian": "Librarian",
    "florist": "Florist", "preschool_teacher": "Preschool Teacher", "nurse": "Nurse",
    "makeup_artist": "Makeup Artist",
}


def occ_stats(rows: list[dict]) -> tuple[float, float]:
    vals = [r["mean"] for r in rows]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


def normalised_curves(data_dir: Path, models: list[str]) -> dict[str, dict[str, tuple]]:
    """{model: {occupation: (layers, mean, s.e.m.)}} of the normalised LALS (occupations present only)."""
    curves = {}
    for model in models:
        occs = json.loads((data_dir / "layer_sweep" / f"{model}.json").read_text())["occupations"]
        curves[model] = {}
        for occ in (o for o in RANKED if o in occs):
            layers = sorted(int(layer) for layer, rows in occs[occ].items() if rows)
            means, sems = zip(*(occ_stats(occs[occ][str(layer)]) for layer in layers))
            curves[model][occ] = (layers, np.array(means) / NORMALISER[model], np.array(sems) / NORMALISER[model])
    return curves


def layer_sweep_figure(data_dir: Path, models: list[str] | None = None) -> plt.Figure:
    """Figure 10: normalised LALS per occupation across layers, one panel per model
    (``models`` restricts it to some of the four models)."""
    panels = [(m, title) for m, title in PANELS if models is None or m in models]
    curves = normalised_curves(data_dir, [m for m, _ in panels])
    colors = {occ: plt.cm.coolwarm(i / (len(RANKED) - 1)) for i, occ in enumerate(RANKED)}
    ncols = 2 if len(panels) > 1 else 1
    nrows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), sharey=True, squeeze=False)
    axes = axes.flatten()
    for ax_idx, (model, title) in enumerate(panels):
        ax = axes[ax_idx]
        for occ, (layers, means, sems) in curves[model].items():
            ax.plot(layers, means, color=colors[occ], marker="o", linewidth=1.8,
                    markersize=4, alpha=0.85, label=OCC_LABELS[occ])
            ax.fill_between(layers, means - sems, means + sems, color=colors[occ], alpha=0.12)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_xticks(layers)
        ax.set_xlim(layers[0] - 1, layers[-1] + 1)
        ax.set_ylim(-1.15, 1.15)
        if ax_idx % ncols == 0:
            ax.set_ylabel("Normalised LALS", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=6, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig
