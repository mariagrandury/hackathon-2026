"""Seaborn-based plotting primitives.

Plots NEVER carry a title — every analysis renders untitled figures and
puts the descriptive text inside the LaTeX caption (`latex_utils.figure_tex`).
This keeps PNGs reusable for both the report and slides without having
to redraw, and avoids title/caption duplication.

Each section's `analyze.py` uses these primitives directly via
matplotlib `GridSpec` to build dense, multi-panel figures. We don't
write one wrapper per output figure — wrappers compose freely in the
section scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ----- theming --------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="notebook",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.titlesize": 12,
        "font.family": "DejaVu Sans",
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    },
)


# ----- palette --------------------------------------------------------------
# Mirrors the Leaderboard / inspect_hf_dataset palette so plots from the
# analysis suite read alike across the project.

C_ENVIADOS = "#3b82f6"        # "sent" — blue
C_VALIDADOS = "#10b981"       # "validated" — green
C_VOTADOS = "#f59e0b"         # "voted" — amber
C_APROBADO = "#22c55e"        # pass / accept
C_PENDIENTE = "#facc15"       # pending
C_RECHAZO = "#ef4444"         # reject
C_NEUTRO = "#9ca3af"
C_HIGHLIGHT = "#8b5cf6"       # purple — accent for one-off detail plots

# 7 validation buckets — reject in warm hues, accept in cool, matching
# the Spanish labels used in the rendered plots.
BUCKET_COLORS = {
    "trivial":    "#fca5a5",
    "stereotype": "#ef4444",
    "unrelated":  "#f97316",
    "knowledge":  "#3b82f6",
    "preference": "#10b981",
    "dynamics":   "#8b5cf6",
    "bias_probe": "#0ea5e9",
}

# Display names (ES) for the validation buckets.
BUCKET_LABELS_ES = {
    "trivial":    "trivial",
    "stereotype": "estereotipo",
    "unrelated":  "sin anclaje",
    "knowledge":  "conocimiento",
    "preference": "preferencia",
    "dynamics":   "dinámica",
    "bias_probe": "trampa de sesgo",
}

VOTE_COLORS = {
    "a":    "#3b82f6",
    "b":    "#f59e0b",
    "both": "#10b981",
    "none": "#9ca3af",
}

VOTE_LABELS_ES = {
    "a":    "A",
    "b":    "B",
    "both": "ambas",
    "none": "ninguna",
}

LANGUAGE_COLORS = {
    "es": "#dc2626",
    "pt": "#16a34a",
    "en": "#2563eb",
}

LANGUAGE_LABELS_ES = {
    "es": "Español",
    "pt": "Portugués",
    "en": "Inglés",
}


# ----- save helpers ---------------------------------------------------------


def save_figure(fig, output_path: str | Path, *, dpi: int = 150) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def setup_figure(figsize: tuple[float, float] = (10.0, 6.0)):
    return plt.subplots(figsize=figsize)


def setup_gridspec(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[float, float] = (12.0, 9.0),
    **gridspec_kw,
):
    """`(fig, gridspec)` ready for `fig.add_subplot(gs[i, j])` composition."""
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(nrows, ncols, **gridspec_kw)
    return fig, gs


# ----- primitives -----------------------------------------------------------
# Thin convenience wrappers around seaborn so section scripts don't repeat
# the same styling boilerplate.


def horizontal_bar(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    palette: Mapping[str, str] | str | None = None,
    hue: str | None = None,
    annotate: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    order: Sequence[str] | None = None,
    legend: bool = True,
) -> None:
    sns.barplot(
        data=df, x=x, y=y, hue=hue, palette=palette,
        ax=ax, order=order, edgecolor="white", linewidth=0.4,
    )
    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()
    if annotate:
        for container in ax.containers:
            ax.bar_label(container, fmt="%g", fontsize=8, padding=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def vertical_bar(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    palette: Mapping[str, str] | str | None = None,
    hue: str | None = None,
    annotate: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    order: Sequence[str] | None = None,
    rotate_xticks: int = 0,
    legend: bool = True,
    reference_line: float | None = None,
) -> None:
    sns.barplot(
        data=df, x=x, y=y, hue=hue, palette=palette,
        ax=ax, order=order, edgecolor="white", linewidth=0.4,
    )
    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()
    if annotate:
        for container in ax.containers:
            ax.bar_label(container, fmt="%g", fontsize=8, padding=2)
    if rotate_xticks:
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotate_xticks)
            tick.set_ha("right")
    if reference_line is not None:
        ax.axhline(reference_line, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def stacked_bar(
    ax,
    df_wide: pd.DataFrame,
    x_col: str,
    stack_cols: Sequence[str],
    palette: Mapping[str, str],
    *,
    xlabel: str = "",
    ylabel: str = "",
    rotate_xticks: int = 30,
    labels: Mapping[str, str] | None = None,
    orientation: str = "vertical",
) -> None:
    if df_wide.empty or not stack_cols:
        ax.text(0.5, 0.5, "(sin datos)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    n = len(df_wide)
    offsets = np.zeros(n, dtype=float)
    labels = labels or {}
    for col in stack_cols:
        vals = df_wide[col].to_numpy(dtype=float)
        label = labels.get(col, col)
        color = palette.get(col, C_NEUTRO)
        if orientation == "horizontal":
            ax.barh(df_wide[x_col], vals, left=offsets, color=color, label=label, edgecolor="white", linewidth=0.4)
        else:
            ax.bar(df_wide[x_col], vals, bottom=offsets, color=color, label=label, edgecolor="white", linewidth=0.4)
        offsets += vals
    if orientation == "vertical":
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if rotate_xticks:
            for tick in ax.get_xticklabels():
                tick.set_rotation(rotate_xticks)
                tick.set_ha("right")
    else:
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    ax.legend(fontsize=8, frameon=False, loc="best")


def heatmap(
    ax,
    data: pd.DataFrame,
    *,
    cmap: str = "Blues",
    annot: bool = True,
    fmt: str = "d",
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str = "",
    ylabel: str = "",
    cbar: bool = True,
    cbar_label: str = "",
) -> None:
    sns.heatmap(
        data, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
        vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
        cbar=cbar, cbar_kws={"label": cbar_label} if cbar_label else None,
        annot_kws={"fontsize": 8},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def donut(
    ax,
    counts: pd.Series,
    *,
    palette: Mapping[str, str] | None = None,
    hole_size: float = 0.55,
    legend_loc: str = "center left",
    legend_bbox: tuple[float, float] = (1.02, 0.5),
) -> None:
    """Donut chart — percentages inside wedges, category names in a side
    legend so many small slices don't crowd labels."""
    counts = counts.sort_values(ascending=False)
    if counts.empty or counts.sum() == 0:
        ax.text(0.5, 0.5, "(sin datos)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    colors = [
        (palette or {}).get(label, sns.color_palette("Set2")[i % 8])
        for i, label in enumerate(counts.index)
    ]
    wedges, _texts, _autos = ax.pie(
        counts.values,
        autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        startangle=90,
        colors=colors,
        wedgeprops={"width": hole_size, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 8, "color": "white", "fontweight": "bold"},
    )
    ax.legend(
        wedges,
        [f"{label}  (n={int(v)})" for label, v in counts.items()],
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        fontsize=8,
        frameon=False,
    )
    ax.set_aspect("equal")


def radial_bar(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    palette: Sequence[str] | None = None,
    value_fmt: str = "{:d}",
    max_value: float | None = None,
) -> None:
    """Polar bar chart — labels around the circle, bar length = value.
    The ax MUST be a polar axes (created via `subplot(projection='polar')`)."""
    n = len(values)
    if n == 0:
        ax.text(0, 0, "(sin datos)", ha="center", va="center")
        return
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.85
    vmax = max_value if max_value is not None else max(values + [1])
    colors = palette or sns.color_palette("Set2", n)
    bars = ax.bar(
        angles, values, width=width, bottom=vmax * 0.10,
        color=colors, edgecolor="white", linewidth=1.2, align="center",
    )
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, vmax * 1.15)
    ax.spines["polar"].set_visible(False)
    ax.grid(alpha=0.3)
    for angle, value, bar in zip(angles, values, bars):
        ax.text(
            angle, bar.get_height() + vmax * 0.13,
            value_fmt.format(int(value)) if "d" in value_fmt else value_fmt.format(value),
            ha="center", va="center", fontsize=9, fontweight="bold",
        )


def lorenz(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    gini: float | None = None,
    color: str = C_ENVIADOS,
    xlabel: str = "proporción acumulada de autores",
    ylabel: str = "proporción acumulada de prompts",
) -> None:
    ax.fill_between(xs, ys, color=color, alpha=0.25)
    ax.plot(xs, ys, color=color, linewidth=2,
            label=f"observado  (Gini = {gini:.2f})" if gini is not None else "observado")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, alpha=0.5, label="igualdad")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def funnel(
    ax,
    labels: Sequence[str],
    counts: Sequence[float],
    *,
    color: str = C_ENVIADOS,
    base: float | None = None,
) -> None:
    n = len(labels)
    if n == 0:
        ax.text(0.5, 0.5, "(sin datos)", ha="center", va="center", transform=ax.transAxes)
        return
    ys = np.arange(n)[::-1]
    counts_arr = np.asarray(counts, dtype=float)
    ax.barh(ys, counts_arr, color=color, height=0.65, edgecolor="white", linewidth=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    denom = base if base is not None and base > 0 else (counts_arr[0] if counts_arr.size and counts_arr[0] > 0 else 1)
    for y, cnt in zip(ys, counts_arr):
        pct = 100.0 * cnt / denom if denom else 0.0
        offset = max(counts_arr.max(), 1) * 0.01
        ax.text(cnt + offset, y, f"{int(cnt)}  ({pct:.0f}%)",
                ha="left", va="center", fontsize=9)
    ax.set_xlabel("prompts")
    ax.set_xlim(0, max(counts_arr.max(), 1) * 1.25)


def errorbar_forest(
    ax,
    df: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    lo_col: str,
    hi_col: str,
    n_col: str | None = None,
    color: str = C_VOTADOS,
    reference: float | None = 0.5,
    xlabel: str = "",
) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "(sin datos)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    sub = df.sort_values(value_col).reset_index(drop=True)
    ys = np.arange(len(sub))
    vals = sub[value_col].to_numpy(dtype=float)
    lows = np.maximum(0.0, vals - sub[lo_col].to_numpy(dtype=float))
    highs = np.maximum(0.0, sub[hi_col].to_numpy(dtype=float) - vals)
    ax.errorbar(vals, ys, xerr=[lows, highs], fmt="o", color=color, ecolor=color, capsize=4, markersize=7)
    if n_col is not None:
        labels = [f"{lbl} (n={int(n)})" for lbl, n in zip(sub[label_col], sub[n_col])]
    else:
        labels = list(sub[label_col])
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    if reference is not None:
        ax.axvline(reference, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel)


def violin(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    palette: Mapping[str, str] | str | None = None,
    order: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    inner: str = "box",
) -> None:
    sns.violinplot(
        data=df, x=x, y=y, hue=x, palette=palette,
        ax=ax, order=order, inner=inner, density_norm="width", legend=False, linewidth=0.6,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def kde(
    ax,
    df: pd.DataFrame,
    *,
    x: str,
    hue: str | None = None,
    palette: Mapping[str, str] | None = None,
    vline: float | None = None,
    vline_label: str | None = None,
    xlabel: str = "",
    fill: bool = True,
) -> None:
    sns.kdeplot(
        data=df, x=x, hue=hue, palette=palette,
        ax=ax, fill=fill, common_norm=False, alpha=0.5, linewidth=1.5,
    )
    if vline is not None:
        ax.axvline(vline, color="black", linestyle="--", linewidth=1.2)
        if vline_label:
            top = ax.get_ylim()[1]
            ax.text(vline, top * 0.95, f"  {vline_label}", ha="left", va="top", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("densidad")


def empty(ax, message: str = "(sin datos)") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color=C_NEUTRO)
    ax.set_axis_off()


# ----- standalone detail-plot helpers ---------------------------------------
# These produce a self-contained figure (one chart per file) saved to disk.
# Use them for the per-category / per-country / per-language breakdowns the
# composite "panorama" + "mapa" figures don't show in isolation.


def detail_horizontal_bar(
    counts: pd.Series,
    output_path: str | Path,
    *,
    color: str = C_ENVIADOS,
    xlabel: str = "n",
    multi_color: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Horizontal bar of value_counts. `multi_color` maps each index value
    to a color; falls back to a single color when None."""
    counts = counts[counts >= 0]
    n_items = max(len(counts), 1)
    fig, ax = plt.subplots(figsize=figsize or (8.5, max(3.0, 0.35 * n_items + 1.2)))
    if counts.empty:
        empty(ax)
        return save_figure(fig, output_path)
    if multi_color is not None:
        palette = [multi_color.get(idx, C_NEUTRO) for idx in counts.index]
        sns.barplot(
            y=list(counts.index), x=counts.values, hue=list(counts.index),
            palette=palette, ax=ax, edgecolor="white", linewidth=0.4, legend=False,
        )
    else:
        sns.barplot(
            y=list(counts.index), x=counts.values, color=color,
            ax=ax, edgecolor="white", linewidth=0.4,
        )
    for container in ax.containers:
        ax.bar_label(container, fmt="%g", fontsize=9, padding=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    return save_figure(fig, output_path)


def detail_vertical_bar(
    counts: pd.Series,
    output_path: str | Path,
    *,
    color: str = C_ENVIADOS,
    ylabel: str = "n",
    rotate_xticks: int = 0,
    reference: float | None = None,
    value_fmt: str = "%g",
    figsize: tuple[float, float] | None = None,
) -> Path:
    n_items = max(len(counts), 1)
    fig, ax = plt.subplots(figsize=figsize or (max(5.5, 0.65 * n_items + 2.0), 4.8))
    if counts.empty:
        empty(ax)
        return save_figure(fig, output_path)
    sns.barplot(
        x=list(counts.index), y=counts.values, color=color,
        ax=ax, edgecolor="white", linewidth=0.4,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt=value_fmt, fontsize=9, padding=2)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if rotate_xticks:
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotate_xticks)
            tick.set_ha("right")
    if reference is not None:
        ax.axhline(reference, color="black", linestyle="--", linewidth=1, alpha=0.5)
    return save_figure(fig, output_path)


def detail_stacked_bar(
    df_wide: pd.DataFrame,
    output_path: str | Path,
    *,
    x_col: str,
    stack_cols: Sequence[str],
    palette: Mapping[str, str],
    ylabel: str = "n",
    labels: Mapping[str, str] | None = None,
    rotate_xticks: int = 30,
    figsize: tuple[float, float] | None = None,
    orientation: str = "vertical",
) -> Path:
    n_items = max(len(df_wide), 1)
    if orientation == "horizontal":
        fig, ax = plt.subplots(figsize=figsize or (8.5, max(3.0, 0.4 * n_items + 1.2)))
    else:
        fig, ax = plt.subplots(figsize=figsize or (max(6.0, 0.7 * n_items + 2.0), 5.0))
    stacked_bar(
        ax, df_wide, x_col=x_col, stack_cols=list(stack_cols),
        palette=palette, ylabel=ylabel, labels=labels,
        rotate_xticks=rotate_xticks, orientation=orientation,
    )
    return save_figure(fig, output_path)


def detail_histogram(
    values: pd.Series,
    output_path: str | Path,
    *,
    bins: int = 30,
    color: str = C_ENVIADOS,
    xlabel: str = "valor",
    vline: float | None = None,
    vline_label: str | None = None,
    group_by: pd.Series | None = None,
    palette: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=figsize or (8.5, 4.8))
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if arr.empty:
        empty(ax)
        return save_figure(fig, output_path)
    if group_by is None:
        sns.histplot(arr, bins=bins, color=color, edgecolor="white", linewidth=0.4, ax=ax)
    else:
        groups = pd.Series(list(group_by)).reset_index(drop=True)
        df_plot = pd.DataFrame({"_v": pd.Series(list(values)).reset_index(drop=True), "_g": groups})
        df_plot = df_plot.dropna(subset=["_v"])
        if df_plot.empty:
            empty(ax)
            return save_figure(fig, output_path)
        sns.histplot(
            data=df_plot, x="_v", hue="_g", bins=bins, multiple="layer",
            palette=palette or LANGUAGE_COLORS, ax=ax,
            edgecolor="white", linewidth=0.3, alpha=0.55,
        )
    if vline is not None:
        ax.axvline(vline, color="black", linestyle="--", linewidth=1.2)
        if vline_label:
            top = ax.get_ylim()[1]
            ax.text(vline, top * 0.95, f"  {vline_label}", ha="left", va="top", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frecuencia")
    return save_figure(fig, output_path)


def detail_lorenz(
    values: Sequence[float],
    output_path: str | Path,
    *,
    gini: float | None = None,
    color: str = C_ENVIADOS,
    figsize: tuple[float, float] = (6.0, 6.0),
) -> Path:
    from . import metrics
    xs, ys = metrics.lorenz_points(values)
    fig, ax = plt.subplots(figsize=figsize)
    lorenz(ax, xs, ys, gini=gini, color=color)
    return save_figure(fig, output_path)


def detail_funnel(
    labels: Sequence[str],
    counts: Sequence[float],
    output_path: str | Path,
    *,
    color: str = C_ENVIADOS,
    base: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> Path:
    n = max(len(labels), 1)
    fig, ax = plt.subplots(figsize=figsize or (8.5, max(3.0, 0.55 * n + 1.5)))
    funnel(ax, labels, counts, color=color, base=base)
    return save_figure(fig, output_path)


def detail_errorbar(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    label_col: str,
    value_col: str,
    lo_col: str,
    hi_col: str,
    n_col: str | None = None,
    color: str = C_VOTADOS,
    reference: float | None = 0.5,
    xlabel: str = "",
    figsize: tuple[float, float] | None = None,
) -> Path:
    n_items = max(len(df), 1)
    fig, ax = plt.subplots(figsize=figsize or (8.5, max(3.0, 0.4 * n_items + 1.5)))
    errorbar_forest(
        ax, df, label_col=label_col, value_col=value_col,
        lo_col=lo_col, hi_col=hi_col, n_col=n_col,
        color=color, reference=reference, xlabel=xlabel,
    )
    return save_figure(fig, output_path)
