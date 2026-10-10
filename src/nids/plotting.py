"""Report figures. One visual language: light surface, recessive axes, a blue sequential ramp for
magnitudes, gray-vs-blue for before/after comparisons, small multiples instead of colour soup."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
BEFORE = "#c3c2b7"        # context series in before/after comparisons
REFERENCE = "#d03b3b"     # threshold reference lines
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
})


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _value_grid(ax, axis: str = "x"):
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)


def class_distribution(counts: pd.Series, path: Path, title: str = "Class distribution") -> Path:
    counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(counts) + 1.2))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.6)
    ax.set_xscale("log")
    total = counts.sum()
    for i, (n) in enumerate(counts.values):
        ax.text(n * 1.15, i, f"{n:,}  ({n / total:.2%})", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(right=counts.max() * 12)
    ax.set_xlabel("Flows (log scale)")
    ax.set_title(title)
    _value_grid(ax)
    return _save(fig, path)


def heatmap(table: pd.DataFrame, path: Path, title: str, fmt: str = "{:.3f}", vmin=None, vmax=None,
            xlabel: str = "", ylabel: str = "") -> Path:
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.35 * table.shape[1] + 2.4, 0.5 * table.shape[0] + 1.4))
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax
    im = ax.imshow(values, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    span = (vmax - vmin) or 1.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                continue
            dark = (v - vmin) / span > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK)
    best = np.unravel_index(np.nanargmax(values), values.shape)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    return _save(fig, path)


def confusion(y_true, y_pred, labels: list[str], path: Path, title: str = "Confusion matrix (row-normalised)") -> Path:
    pred_labels = labels + [p for p in sorted(set(map(str, y_pred))) if p not in labels]
    cm = confusion_matrix(y_true, y_pred, labels=pred_labels)[: len(labels)]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n_rows, n_cols = norm.shape
    fig, ax = plt.subplots(figsize=(0.75 * n_cols + 2.6, 0.6 * n_rows + 1.6))
    ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_cols), pred_labels, rotation=35, ha="right")
    ax.set_yticks(range(n_rows), labels)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(n_rows):
        for j in range(n_cols):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{norm[i, j]:.0%}" if norm[i, j] >= 0.005 else "<1%", ha="center", va="center",
                    fontsize=8, color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, path)


def before_after(before: pd.Series, after: pd.Series, path: Path, title: str, before_label: str,
                 after_label: str, xlabel: str) -> Path:
    """Paired horizontal bars per class: gray = before/baseline, blue = after/ours."""
    classes = list(after.index)
    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(classes) + 1.4))
    h = 0.36
    before_vals = before.reindex(classes).fillna(0).values
    # y axis is inverted: the "before" bar sits above the "after" bar, matching the legend order.
    ax.barh(y - h / 2 - 0.01, before_vals, height=h, color=BEFORE, label=before_label)
    ax.barh(y + h / 2 + 0.01, after.values, height=h, color=BLUE, label=after_label)
    for i, (b, a) in enumerate(zip(before_vals, after.values)):
        ax.text(b + 0.01, i - h / 2 - 0.01, f"{b:.2f}", va="center", fontsize=8, color=MUTED)
        ax.text(a + 0.01, i + h / 2 + 0.01, f"{a:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=9, handlelength=1.2)
    _value_grid(ax)
    return _save(fig, path)


def pr_curves(y_true, proba: np.ndarray, classes, path: Path, benign: str) -> Path:
    """Small multiples: one precision-recall curve per attack family."""
    y_true = np.asarray(y_true).astype(str)
    attack = [(j, c) for j, c in enumerate(classes) if c != benign and (y_true == c).any()]
    cols = 4
    rows = int(np.ceil(len(attack) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False, sharex=True, sharey=True)
    for ax in axes.flat[len(attack):]:
        ax.set_visible(False)
    for ax, (j, c) in zip(axes.flat, attack):
        positive = y_true == c
        precision, recall, _ = precision_recall_curve(positive, proba[:, j])
        ap = average_precision_score(positive, proba[:, j])
        ax.plot(recall, precision, color=BLUE, linewidth=2)
        ax.axhline(positive.mean(), color=AXIS, linewidth=1, linestyle="--")
        ax.set_title(f"{c}  ·  AP {ap:.2f}", fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        _value_grid(ax, "both")
    for ax in axes[-1]:
        ax.set_xlabel("Recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision")
    fig.suptitle("Precision-recall per attack family (dashed = chance level)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def importance(imp: pd.DataFrame, path: Path, top: int = 15) -> Path:
    imp = imp.sort_values("importance", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(imp) + 1.2))
    ax.barh(imp.index, imp["importance"], xerr=imp.get("std"), color=BLUE, height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0})
    ax.set_xlabel("Drop in validation macro-F1 when the feature is shuffled")
    ax.set_title(f"Top {len(imp)} features (permutation importance)")
    _value_grid(ax)
    return _save(fig, path)


def psi_bars(psi: pd.Series, threshold: float, path: Path, top: int = 15,
             title: str = "Feature drift between past and recent traffic") -> Path:
    psi = psi.sort_values(ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(psi) + 1.2))
    ax.barh(psi.index, psi.values, color=[BLUE if v >= threshold else BEFORE for v in psi.values], height=0.6)
    ax.axvline(threshold, color=REFERENCE, linewidth=1, linestyle="--")
    ax.text(threshold, len(psi) - 0.4, f"  drift alert ({threshold})", color=REFERENCE, fontsize=8, va="bottom")
    ax.set_xlabel("Population Stability Index (blue = above the drift-alert line)")
    ax.set_title(title)
    _value_grid(ax)
    return _save(fig, path)
