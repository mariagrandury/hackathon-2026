"""Análisis de la validación de prompts — sección 03.

  1. `panorama_validacion.png` (HEADLINE) — composición 2×2.
  2. `mapa_pais_bucket.png` (HEADLINE) — heatmap país × bucket.
  3. `validacion.csv` + `validacion.tex` — resumen por bucket.
  4. `plots/detalle/*.png` + `.tex` — embudo de slots, distribución de
     buckets, buckets por idioma, tasa de aceptación por país,
     top validadores, tasa de unanimidad por idioma, acuerdo
     aceptación/rechazo. Cobertura completa de las columnas
     `prompt_validation_{1,2,3}` del dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import data as _data
from analysis._common import data_loading as dl
from analysis._common import io_utils, latex_utils, metrics, plotting as P

import matplotlib.pyplot as plt
import seaborn as sns


# ----- plot 1: panorama de validación --------------------------------------


def plot_panorama_validacion(flat: pd.DataFrame, long: pd.DataFrame, out: Path) -> Path:
    fig = plt.figure(figsize=(16.0, 12.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.40)

    # --- Panel A: barra polar por bucket ---
    ax_a = fig.add_subplot(gs[0, 0], projection="polar")
    if long.empty:
        ax_a.text(0, 0, "(sin validaciones)", ha="center", va="center")
        ax_a.set_axis_off()
    else:
        counts = long["choice"].value_counts().reindex(_data.VALIDATION_CHOICES, fill_value=0)
        labels = [P.BUCKET_LABELS_ES[k] for k in counts.index]
        colors = [P.BUCKET_COLORS[k] for k in counts.index]
        P.radial_bar(ax_a, labels, counts.values.tolist(), palette=colors)

    # --- Panel B: embudo de slots llenos por prompt ---
    ax_b = fig.add_subplot(gs[0, 1])
    if flat.empty:
        P.empty(ax_b)
    else:
        slot_counts = flat["validation_status"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
        labels = ["0 slots", "1 slot", "2 slots", "3 slots\n(totalmente validado)"]
        sns.barplot(
            x=labels, y=slot_counts.values,
            hue=labels, palette=[P.C_NEUTRO, "#fde047", P.C_PENDIENTE, P.C_VALIDADOS],
            ax=ax_b, edgecolor="white", linewidth=0.4, legend=False,
        )
        for container in ax_b.containers:
            ax_b.bar_label(container, fmt="%d", fontsize=9, padding=2)
        ax_b.set_xlabel("")
        ax_b.set_ylabel("número de prompts")

    # --- Panel C: donut aceptación / rechazo ---
    ax_c = fig.add_subplot(gs[1, 0])
    if long.empty:
        P.empty(ax_c)
    else:
        sides = long["side"].value_counts()
        side_labels = {"accept": "aceptación", "reject": "rechazo", "": "sin lado"}
        side_palette = {"aceptación": P.C_APROBADO, "rechazo": P.C_RECHAZO, "sin lado": P.C_NEUTRO}
        sides.index = sides.index.map(lambda s: side_labels.get(s, s))
        P.donut(ax_c, sides, palette=side_palette, hole_size=0.50,
                legend_loc="center left", legend_bbox=(1.0, 0.5))

    # --- Panel D: tasa de unanimidad por idioma ---
    ax_d = fig.add_subplot(gs[1, 1])
    if long.empty:
        P.empty(ax_d)
    else:
        unan = metrics.unanimity_rate_by_group(long, group_col="prompt_language")
        if unan.empty or unan["n_full"].sum() == 0:
            P.empty(ax_d, message="(aún no hay prompts con los 3 slots llenos)")
        else:
            unan = unan.assign(
                idioma=unan["prompt_language"].map(P.LANGUAGE_LABELS_ES).fillna(unan["prompt_language"]),
                pct=(unan["unanimity_rate"] * 100).round(1),
                label=lambda d: d["idioma"] + "\n(n_full=" + d["n_full"].astype(int).astype(str) + ")",
            )
            palette = {row["label"]: P.LANGUAGE_COLORS.get(row["prompt_language"], P.C_NEUTRO) for _, row in unan.iterrows()}
            sns.barplot(
                data=unan, x="label", y="pct", hue="label",
                palette=palette, ax=ax_d, edgecolor="white", linewidth=0.4, legend=False,
            )
            for container in ax_d.containers:
                ax_d.bar_label(container, fmt="%.1f", fontsize=9, padding=2)
            ax_d.set_xlabel("")
            ax_d.set_ylabel("% prompts con los 3 validadores de acuerdo")
            ax_d.set_ylim(0, 105)
            ax_d.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.5)

    return P.save_figure(fig, out / "panorama_validacion.png")


# ----- plot 2: mapa país × bucket ------------------------------------------


def plot_mapa_pais_bucket(long: pd.DataFrame, out: Path) -> Path:
    out_path = out / "mapa_pais_bucket.png"
    fig, ax = plt.subplots(figsize=(11.0, 7.5), facecolor="white")
    if long.empty:
        P.empty(ax)
        return P.save_figure(fig, out_path)

    pivot = (
        long.groupby(["prompt_country_display", "choice"]).size()
        .unstack(fill_value=0)
        .reindex(columns=list(_data.VALIDATION_CHOICES), fill_value=0)
    )
    # Ordenar países por volumen total y renombrar columnas a ES.
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot.columns = [P.BUCKET_LABELS_ES[c] for c in pivot.columns]
    pivot.index.name = None

    sns.heatmap(
        pivot, annot=True, fmt="d", cmap="YlGnBu",
        linewidths=0.6, linecolor="white", ax=ax,
        cbar_kws={"label": "validaciones registradas"},
        annot_kws={"fontsize": 8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    return P.save_figure(fig, out_path)


# ----- summary table -------------------------------------------------------


def build_summary(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=[
            "Bucket", "Lado", "n validaciones", "% del total",
        ])
    counts = long["choice"].value_counts().reindex(_data.VALIDATION_CHOICES, fill_value=0)
    total = int(counts.sum()) if counts.sum() > 0 else 1
    rows = []
    for choice in _data.VALIDATION_CHOICES:
        n = int(counts.get(choice, 0))
        side = "aceptación" if choice in _data.ACCEPT_VALIDATION_CHOICES else "rechazo"
        rows.append({
            "Bucket": P.BUCKET_LABELS_ES[choice],
            "Lado": side,
            "n validaciones": n,
            "% del total": round(n / total * 100, 1),
        })
    return pd.DataFrame(rows)


# ----- detail plots ---------------------------------------------------------


def _emit_detail(fig_path: Path, label: str, caption: str) -> Path:
    latex_utils.save_figure_tex(fig_path, caption=caption, label=label)
    return fig_path


def emit_detail_plots(flat: pd.DataFrame, long: pd.DataFrame, detalle_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if flat.empty:
        return outputs

    # --- embudo de slots ocupados ---
    state_counts = flat["validation_status"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    state_counts.index = ["0 slots", "1 slot", "2 slots", "3 slots\n(totalmente validado)"]
    path = P.detail_vertical_bar(
        state_counts, detalle_dir / "embudo_slots_validacion.png",
        color=P.C_VALIDADOS, ylabel="prompts",
    )
    outputs.append(_emit_detail(path, "embudo_slots_validacion",
        f"Número de prompts según el número de slots de validación "
        f"ocupados, sobre el total de $n={len(flat)}$ prompts."
    ))

    if long.empty:
        return outputs

    # --- distribución de buckets ---
    bucket_counts = (
        long["choice"].value_counts()
        .reindex(_data.VALIDATION_CHOICES, fill_value=0)
    )
    bucket_counts.index = [P.BUCKET_LABELS_ES[c] for c in bucket_counts.index]
    multi_color = {P.BUCKET_LABELS_ES[k]: v for k, v in P.BUCKET_COLORS.items()}
    path = P.detail_horizontal_bar(
        bucket_counts, detalle_dir / "distribucion_buckets.png",
        xlabel="validaciones", multi_color=multi_color,
    )
    outputs.append(_emit_detail(path, "distribucion_buckets",
        f"Distribución absoluta de las $n={len(long)}$ validaciones "
        f"emitidas entre los siete buckets de la taxonomía. Los tres "
        f"primeros son de rechazo y los cuatro siguientes de aceptación."
    ))

    # --- buckets por idioma del prompt (stacked) ---
    tab = (
        long.groupby(["prompt_language", "choice"]).size()
        .unstack(fill_value=0)
        .reindex(columns=list(_data.VALIDATION_CHOICES), fill_value=0)
    )
    tab.index = [P.LANGUAGE_LABELS_ES.get(c, c) for c in tab.index]
    tab.index.name = "idioma"
    tab.columns = [P.BUCKET_LABELS_ES[c] for c in tab.columns]
    tab_wide = tab.reset_index()
    palette_bucket = {P.BUCKET_LABELS_ES[k]: v for k, v in P.BUCKET_COLORS.items()}
    path = P.detail_stacked_bar(
        tab_wide, detalle_dir / "buckets_por_idioma.png",
        x_col="idioma", stack_cols=list(tab.columns),
        palette=palette_bucket, ylabel="validaciones",
        rotate_xticks=0,
    )
    outputs.append(_emit_detail(path, "buckets_por_idioma",
        f"Composición de buckets dentro de cada idioma de la interfaz. "
        f"Permite detectar diferencias entre poblaciones lingüísticas "
        f"en qué dimensiones culturales están dispuestas a validar."
    ))

    # --- buckets por país (stacked) ---
    tab_pais = (
        long.groupby(["prompt_country_display", "choice"]).size()
        .unstack(fill_value=0)
        .reindex(columns=list(_data.VALIDATION_CHOICES), fill_value=0)
    )
    tab_pais = tab_pais.loc[tab_pais.sum(axis=1).sort_values(ascending=False).index]
    tab_pais.index.name = "pais"
    tab_pais.columns = [P.BUCKET_LABELS_ES[c] for c in tab_pais.columns]
    tab_wide_pais = tab_pais.reset_index()
    path = P.detail_stacked_bar(
        tab_wide_pais, detalle_dir / "buckets_por_pais.png",
        x_col="pais", stack_cols=list(tab_pais.columns),
        palette=palette_bucket, ylabel="validaciones",
        orientation="horizontal",
    )
    outputs.append(_emit_detail(path, "buckets_por_pais",
        f"Composición de buckets dentro de cada país, ordenado de mayor "
        f"a menor volumen. Sirve para ver qué dimensiones culturales se "
        f"están validando con más fuerza en cada región."
    ))

    # --- tasa de aceptación por país ---
    grouped = (
        flat.groupby("country_display")
        .agg(total=("id", "size"), fully=("is_fully_validated", "sum"))
    )
    grouped = grouped[grouped["total"] >= 5]
    if not grouped.empty:
        grouped["rate"] = (grouped["fully"] / grouped["total"] * 100).round(1)
        grouped = grouped.sort_values("rate", ascending=False)
        labels = [f"{idx} (n={int(row.total)})" for idx, row in grouped.iterrows()]
        rate_series = pd.Series(grouped["rate"].values, index=labels)
        path = P.detail_horizontal_bar(
            rate_series, detalle_dir / "tasa_aceptacion_por_pais.png",
            color=P.C_APROBADO, xlabel="% totalmente validados",
        )
        outputs.append(_emit_detail(path, "tasa_aceptacion_por_pais",
            f"Porcentaje de prompts que alcanzaron el estado "
            f"\\emph{{totalmente validado}} (tres etiquetas de aceptación) "
            f"por país, contando sólo países con $n \\geq 5$ prompts para "
            f"que la cola larga no introduzca ruido."
        ))

    # --- top validadores ---
    counts = long["validator_username"].value_counts().head(20)
    path = P.detail_horizontal_bar(
        counts, detalle_dir / "top_validadores.png",
        color=P.C_VALIDADOS, xlabel="validaciones",
    )
    outputs.append(_emit_detail(path, "top_validadores",
        f"Los 20 validadores más activos. Cada cuenta es el número de "
        f"slots de validación que ese usuario ha llenado en cualquiera "
        f"de los tres slots por prompt."
    ))

    # --- tasa de unanimidad por idioma (standalone) ---
    unan = metrics.unanimity_rate_by_group(long, group_col="prompt_language")
    if not unan.empty and unan["n_full"].sum() > 0:
        unan = unan.assign(idioma=unan["prompt_language"].map(P.LANGUAGE_LABELS_ES).fillna(unan["prompt_language"]))
        ser = pd.Series(
            (unan["unanimity_rate"] * 100).round(1).values,
            index=[f"{lang} (n_full={int(n)})" for lang, n in zip(unan["idioma"], unan["n_full"])]
        )
        path = P.detail_vertical_bar(
            ser, detalle_dir / "tasa_unanimidad_por_idioma.png",
            color=P.C_HIGHLIGHT, ylabel="% prompts unanimes",
            value_fmt="%.1f", reference=100.0,
        )
        outputs.append(_emit_detail(path, "tasa_unanimidad_por_idioma",
            f"Sobre los prompts con los tres slots de validación llenos, "
            f"porcentaje en el que los tres validadores eligieron "
            f"exactamente el mismo bucket, segmentado por idioma del "
            f"prompt. Métrica de fiabilidad simple — la sustituimos por "
            f"Cohen's kappa intencionalmente porque el emparejamiento "
            f"de validadores no es sistemático en el pipeline."
        ))

    # --- acuerdo aceptación / rechazo (standalone) ---
    ar = metrics.accept_reject_agreement(long)
    if ar["n_prompts"] > 0:
        ser = pd.Series([ar["rate"] * 100], index=[f"acordaron lado\n(n={ar['n_prompts']} prompts)"])
        path = P.detail_vertical_bar(
            ser, detalle_dir / "acuerdo_aceptacion_rechazo.png",
            color=P.C_APROBADO, ylabel="%",
            value_fmt="%.1f", reference=100.0,
            figsize=(5.5, 4.8),
        )
        outputs.append(_emit_detail(path, "acuerdo_aceptacion_rechazo",
            f"Sobre los prompts con $\\geq 2$ validaciones, porcentaje "
            f"en el que todos los validadores eligieron buckets del "
            f"mismo lado (aceptación o rechazo), incluso si dentro de "
            f"ese lado discreparon en el bucket exacto. Una tasa alta "
            f"con baja unanimidad indica que la taxonomía de 7 buckets "
            f"tiene ambigüedad interna, pero el juicio binario "
            f"aceptación/rechazo está sólido."
        ))

    return outputs


# ----- entry points ---------------------------------------------------------


def run(
    participants_df: pd.DataFrame | None = None,
    prompts_df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    out_dir = Path(out_dir or io_utils.section_dir(__file__))
    plots_dir = io_utils.ensure_dir(out_dir / "plots")
    detalle_dir = io_utils.ensure_dir(plots_dir / "detalle")

    flat = dl.load_prompts_flat(prompts_df)
    long = dl.load_validations_long(prompts_df)
    summary = build_summary(long)

    csv_path = io_utils.save_csv(summary, out_dir / "validacion.csv")
    n_val = len(long)
    n_validators = long["validator_username"].nunique() if not long.empty else 0
    ar_stats = metrics.accept_reject_agreement(long) if not long.empty else {"rate": 0.0, "n_prompts": 0}
    table_caption = (
        f"Distribución de las $n={n_val}$ validaciones registradas por "
        f"$n={n_validators}$ validadores únicos, agrupadas por bucket. "
        f"Los buckets se dividen en tres de \\textit{{rechazo}} (trivial, "
        f"estereotipo, sin anclaje) y cuatro de \\textit{{aceptación}} "
        f"alineados con las dimensiones culturales de AlKhamissi et al. "
        f"(2025): conocimiento, preferencia, dinámica, trampa de sesgo. "
        f"En prompts con al menos dos validaciones, los "
        f"validadores coincidieron en el lado aceptación/rechazo en el "
        f"{ar_stats['rate'] * 100:.1f}\\% de los casos "
        f"(sobre $n={ar_stats['n_prompts']}$ prompts)."
    )
    latex_utils.save_table_tex(csv_path, summary, table_caption, label="validacion")
    print(f"wrote {csv_path}  ({len(summary)} filas)")

    panorama_path = plot_panorama_validacion(flat, long, plots_dir)
    n_prompts_fully = int(flat["is_fully_validated"].sum()) if not flat.empty else 0
    latex_utils.save_figure_tex(
        panorama_path,
        caption=(
            f"Panorama de la validación de prompts. Panel superior "
            f"izquierdo: barras polares con la distribución absoluta de "
            f"las $n={n_val}$ validaciones registradas entre los siete "
            f"buckets de la taxonomía (los tres de rechazo en tonos "
            f"cálidos, los cuatro de aceptación en tonos fríos). Panel "
            f"superior derecho: distribución de prompts según el número "
            f"de slots de validación ocupados (0, 1, 2 o 3 de 3). Panel "
            f"inferior izquierdo: gráfico de dona con el reparto "
            f"agregado entre validaciones de aceptación y de rechazo. "
            f"Panel inferior derecho: tasa de unanimidad por idioma — "
            f"qué porcentaje de los prompts totalmente validados "
            f"recibieron el mismo bucket de los tres validadores. "
            f"En este corte, $n={n_prompts_fully}$ prompts tienen los "
            f"tres slots llenos."
        ),
        label="panorama_validacion",
    )
    print(f"wrote {panorama_path}")

    mapa_path = plot_mapa_pais_bucket(long, plots_dir)
    latex_utils.save_figure_tex(
        mapa_path,
        caption=(
            f"Validaciones registradas por país y bucket. Cada celda "
            f"muestra cuántas etiquetas de validación se han emitido para "
            f"prompts originados en ese país, separadas por el bucket "
            f"elegido. Los países están ordenados de mayor a menor "
            f"volumen de validaciones. La intensidad del color resalta "
            f"qué dimensiones culturales están concentrando la atención "
            f"de los validadores en cada región, lo que sirve para "
            f"detectar tanto sesgos de cobertura geográfica como sesgos "
            f"de dimensión cultural dentro de un país."
        ),
        label="mapa_pais_bucket",
    )
    print(f"wrote {mapa_path}")

    detail_paths = emit_detail_plots(flat, long, detalle_dir)
    for p in detail_paths:
        print(f"wrote {p}")

    return [csv_path, panorama_path, mapa_path, *detail_paths]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
