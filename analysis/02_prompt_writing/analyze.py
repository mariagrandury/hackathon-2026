"""Análisis de la escritura de prompts — sección 02.

  1. `panorama_prompts.png` (HEADLINE) — composición 2×2.
  2. `mapa_pais_idioma_estado.png` (HEADLINE) — heatmaps duales.
  3. `prompts.csv` + `prompts.tex` — resumen por país.
  4. `plots/detalle/*.png` + `.tex` — embudo del pipeline, prompts por
     país, prompts por idioma, distribución por idioma cruzado, longitud
     del prompt, uso de system prompt, top autores, histograma de
     prompts por autor, curva de Lorenz. Cobertura completa de la
     información sobre los prompts disponibles en el dataset
     `cultural_preferences`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis._common import data_loading as dl
from analysis._common import io_utils, latex_utils, metrics, plotting as P

import matplotlib.pyplot as plt
import seaborn as sns


# ----- plot 1: panorama de prompts ------------------------------------------


def plot_panorama_prompts(flat: pd.DataFrame, out: Path) -> Path:
    fig = plt.figure(figsize=(16.0, 12.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32)

    # --- Panel A: embudo del ciclo de vida ---
    ax_a = fig.add_subplot(gs[0, 0])
    if flat.empty:
        P.empty(ax_a)
    else:
        written = len(flat)
        fully_val = int(flat["is_fully_validated"].sum())
        has_ans = int(flat["has_answers"].sum())
        fully_voted = int((flat["n_votes"] >= 3).sum())
        v0 = int(flat["is_v0_import"].sum())
        labels = [
            "escritos",
            "totalmente\nvalidados",
            f"con respuestas\n(a+b)",
            "totalmente\nvotados",
        ]
        P.funnel(ax_a, labels, [written, fully_val, has_ans, fully_voted],
                 color=P.C_ENVIADOS, base=written)
        ax_a.text(
            0.99, 1.02,
            f"de los $n={written}$ prompts, $v0$-importados = {v0}",
            transform=ax_a.transAxes, ha="right", va="bottom", fontsize=8, color=P.C_NEUTRO,
        )

    # --- Panel B: curva de Lorenz + Gini ---
    ax_b = fig.add_subplot(gs[0, 1])
    real_authors = flat[~flat["is_v0_import"]] if not flat.empty else flat
    per_author = (
        real_authors["username"].value_counts().values
        if not real_authors.empty else np.array([])
    )
    if per_author.size == 0:
        P.empty(ax_b)
    else:
        xs, ys = metrics.lorenz_points(per_author)
        gini_value = metrics.gini(per_author)
        P.lorenz(ax_b, xs, ys, gini=gini_value, color=P.C_ENVIADOS)

    # --- Panel C: violinplot de longitud por idioma ---
    ax_c = fig.add_subplot(gs[1, 0])
    if flat.empty:
        P.empty(ax_c)
    else:
        len_df = flat[["prompt_chars", "language"]].copy()
        # Cortar la cola larga para no aplastar el cuerpo del violín.
        cap = float(len_df["prompt_chars"].quantile(0.97))
        len_df = len_df[(len_df["prompt_chars"] > 0) & (len_df["prompt_chars"] <= cap)]
        len_df["idioma"] = len_df["language"].map(P.LANGUAGE_LABELS_ES).fillna(len_df["language"])
        palette_es = {P.LANGUAGE_LABELS_ES[k]: v for k, v in P.LANGUAGE_COLORS.items() if k in P.LANGUAGE_LABELS_ES}
        sns.violinplot(
            data=len_df, x="idioma", y="prompt_chars", hue="idioma",
            palette=palette_es, ax=ax_c, inner="quartile",
            density_norm="width", legend=False, linewidth=0.7,
        )
        ax_c.set_xlabel("")
        ax_c.set_ylabel(f"caracteres del prompt  (cota = pct. 97 = {cap:.0f})")

    # --- Panel D: top autores (sin v0) ---
    ax_d = fig.add_subplot(gs[1, 1])
    if real_authors.empty:
        P.empty(ax_d)
    else:
        top = real_authors["username"].value_counts().head(15).reset_index()
        top.columns = ["autor", "n_prompts"]
        sns.barplot(
            data=top, y="autor", x="n_prompts", color=P.C_ENVIADOS,
            ax=ax_d, edgecolor="white", linewidth=0.4,
        )
        for container in ax_d.containers:
            ax_d.bar_label(container, fmt="%d", fontsize=8, padding=2)
        ax_d.set_xlabel("prompts escritos")
        ax_d.set_ylabel("")

    return P.save_figure(fig, out / "panorama_prompts.png")


# ----- plot 2: mapa país × idioma × estado --------------------------------


def plot_mapa_pais_idioma_estado(flat: pd.DataFrame, out: Path) -> Path:
    """Cuadrícula con dos paneles lado a lado:
       - izquierda: heatmap país × idioma con conteo de prompts
       - derecha: heatmap país × estado de validación (0/1/2/3 slots) con %
    Le permite a un organizador ver, de un vistazo, dónde tenemos prompts
    y cuánto les falta para llegar a fase de votación."""
    fig = plt.figure(figsize=(13.5, 7.0), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.30, width_ratios=[1, 1.15])

    # --- izquierda: país × idioma (count) ---
    ax_l = fig.add_subplot(gs[0, 0])
    if flat.empty:
        P.empty(ax_l)
    else:
        pivot_lang = (
            flat.groupby(["country_display", "language"]).size()
            .unstack(fill_value=0)
            .rename(columns=P.LANGUAGE_LABELS_ES)
        )
        pivot_lang = pivot_lang.loc[pivot_lang.sum(axis=1).sort_values(ascending=False).index]
        sns.heatmap(
            pivot_lang, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, linecolor="white", ax=ax_l,
            cbar_kws={"label": "prompts"}, annot_kws={"fontsize": 8},
        )
        ax_l.set_xlabel("idioma de la interfaz")
        ax_l.set_ylabel("")
        for tick in ax_l.get_xticklabels():
            tick.set_rotation(0)

    # --- derecha: país × estado de validación (%) ---
    ax_r = fig.add_subplot(gs[0, 1])
    if flat.empty:
        P.empty(ax_r)
    else:
        pivot_val = (
            flat.groupby(["country_display", "validation_status"]).size()
            .unstack(fill_value=0)
            .reindex(columns=[0, 1, 2, 3], fill_value=0)
        )
        pivot_val = pivot_val.loc[pivot_val.sum(axis=1).sort_values(ascending=False).index]
        # Normalizar por país para que países con pocos prompts no se vean
        # vacíos — el % es más comparable.
        row_sum = pivot_val.sum(axis=1).replace(0, 1)
        pct = pivot_val.div(row_sum, axis=0).mul(100).round(0)
        pct.columns = ["0 validaciones", "1 validación", "2 validaciones", "3 validaciones"]
        sns.heatmap(
            pct, annot=True, fmt=".0f", cmap="rocket_r",
            linewidths=0.5, linecolor="white", ax=ax_r,
            cbar_kws={"label": "% de prompts del país"},
            annot_kws={"fontsize": 8}, vmin=0, vmax=100,
        )
        ax_r.set_xlabel("estado de validación")
        ax_r.set_ylabel("")
        for tick in ax_r.get_xticklabels():
            tick.set_rotation(15)
            tick.set_ha("right")

    return P.save_figure(fig, out / "mapa_pais_idioma_estado.png")


# ----- summary table -------------------------------------------------------


def build_summary(flat: pd.DataFrame) -> pd.DataFrame:
    if flat.empty:
        return pd.DataFrame(columns=[
            "País", "Prompts", "Autores únicos", "v0 (importados)",
            "Longitud media (carac.)", "% con system prompt",
            "% totalmente validados",
        ])
    grouped = flat.groupby("country_display", dropna=False)
    rows: list[dict] = []
    for country, sub in grouped:
        rows.append({
            "País": country if country else "(sin dato)",
            "Prompts": len(sub),
            "Autores únicos": sub["username"].nunique(),
            "v0 (importados)": int(sub["is_v0_import"].sum()),
            "Longitud media (carac.)": round(float(sub["prompt_chars"].mean()), 1),
            "% con system prompt": round(float(sub["has_system_prompt"].mean() * 100), 1),
            "% totalmente validados": round(float(sub["is_fully_validated"].mean() * 100), 1),
        })
    return pd.DataFrame(rows).sort_values("Prompts", ascending=False).reset_index(drop=True)


# ----- detail plots ---------------------------------------------------------


def _emit_detail(fig_path: Path, label: str, caption: str) -> Path:
    latex_utils.save_figure_tex(fig_path, caption=caption, label=label)
    return fig_path


def emit_detail_plots(flat: pd.DataFrame, detalle_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if flat.empty:
        return outputs
    n = len(flat)
    v0 = int(flat["is_v0_import"].sum())

    # --- embudo del pipeline (versión standalone) ---
    written = n
    fully_val = int(flat["is_fully_validated"].sum())
    has_ans = int(flat["has_answers"].sum())
    fully_voted = int((flat["n_votes"] >= 3).sum())
    path = P.detail_funnel(
        ["escritos", "totalmente validados", "con respuestas (a+b)", "totalmente votados"],
        [written, fully_val, has_ans, fully_voted],
        detalle_dir / "embudo_pipeline.png",
        color=P.C_ENVIADOS, base=written,
    )
    outputs.append(_emit_detail(path, "embudo_pipeline",
        f"Embudo del ciclo de vida de los $n={n}$ prompts en el dataset "
        f"\\texttt{{cultural\\_preferences}} (incluye los {v0} prompts "
        f"importados con sentinel \\texttt{{v0}}). Cada barra cuenta los "
        f"prompts en ese estado independientemente — el estado "
        f"\\textit{{con respuestas}} no requiere validación porque los "
        f"prompts v0 entran al pipeline con par $(a,b)$ pre-cargado."
    ))

    # --- prompts por país ---
    counts = flat["country_display"].fillna("?").value_counts()
    path = P.detail_horizontal_bar(
        counts, detalle_dir / "prompts_por_pais.png",
        color=P.C_ENVIADOS, xlabel="prompts",
    )
    outputs.append(_emit_detail(path, "prompts_por_pais",
        f"Distribución de los $n={n}$ prompts por país de origen del "
        f"autor, contando los prompts v0 importados ({v0})."
    ))

    # --- prompts por idioma ---
    counts = (
        flat["language"].fillna("?")
        .map(P.LANGUAGE_LABELS_ES).fillna(flat["language"].fillna("?"))
        .value_counts()
    )
    path = P.detail_vertical_bar(
        counts, detalle_dir / "prompts_por_idioma.png",
        color=P.C_VALIDADOS, ylabel="prompts",
    )
    outputs.append(_emit_detail(path, "prompts_por_idioma",
        f"Distribución de los $n={n}$ prompts por idioma de la interfaz "
        f"en la que el autor estaba registrado."
    ))

    # --- prompts país × idioma ---
    tab = (
        flat.groupby(["country_display", "language"]).size()
        .unstack(fill_value=0)
    )
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    tab.columns = [P.LANGUAGE_LABELS_ES.get(c, c) for c in tab.columns]
    palette_lang = {P.LANGUAGE_LABELS_ES[k]: v for k, v in P.LANGUAGE_COLORS.items()}
    tab_wide = tab.reset_index()
    path = P.detail_stacked_bar(
        tab_wide, detalle_dir / "prompts_pais_x_idioma.png",
        x_col="country_display", stack_cols=list(tab.columns),
        palette=palette_lang, ylabel="prompts",
        orientation="horizontal",
    )
    outputs.append(_emit_detail(path, "prompts_pais_x_idioma",
        f"Composición lingüística de los prompts escritos por país. "
        f"Los prompts v0 (n={v0}) están todos en español por construcción "
        f"del importador \\texttt{{import\\_dpo\\_pairs.py}}."
    ))

    # --- longitud del prompt (caracteres) por idioma ---
    cap = float(flat["prompt_chars"].quantile(0.97))
    sub_len = flat[(flat["prompt_chars"] > 0) & (flat["prompt_chars"] <= cap)]
    path = P.detail_histogram(
        sub_len["prompt_chars"], detalle_dir / "longitud_prompt_caracteres.png",
        bins=30, xlabel="caracteres del prompt",
        group_by=sub_len["language"].map(P.LANGUAGE_LABELS_ES).fillna(sub_len["language"]),
        palette={P.LANGUAGE_LABELS_ES[k]: v for k, v in P.LANGUAGE_COLORS.items()},
    )
    outputs.append(_emit_detail(path, "longitud_prompt_caracteres",
        f"Distribución de la longitud del prompt en caracteres, "
        f"segmentada por idioma. La cota superior se fija en el percentil "
        f"97 ($\\approx {cap:.0f}$ caracteres) para que la cola larga "
        f"no aplaste el cuerpo de la distribución."
    ))

    # --- longitud del prompt (palabras) ---
    cap_w = float(flat["prompt_words"].quantile(0.97))
    sub_w = flat[(flat["prompt_words"] > 0) & (flat["prompt_words"] <= cap_w)]
    path = P.detail_histogram(
        sub_w["prompt_words"], detalle_dir / "longitud_prompt_palabras.png",
        bins=30, xlabel="palabras del prompt",
        group_by=sub_w["language"].map(P.LANGUAGE_LABELS_ES).fillna(sub_w["language"]),
        palette={P.LANGUAGE_LABELS_ES[k]: v for k, v in P.LANGUAGE_COLORS.items()},
    )
    outputs.append(_emit_detail(path, "longitud_prompt_palabras",
        f"Distribución del número de palabras por prompt, segmentado por "
        f"idioma. Cota superior fijada en el percentil 97 "
        f"($\\approx {cap_w:.0f}$ palabras)."
    ))

    # --- uso de system prompt por idioma ---
    rate = (
        flat.groupby("language")["has_system_prompt"]
        .mean().mul(100).round(1).sort_index()
    )
    counts_lang = flat.groupby("language").size()
    rate.index = [f"{P.LANGUAGE_LABELS_ES.get(k, k)} (n={int(counts_lang[k])})" for k in rate.index]
    path = P.detail_vertical_bar(
        rate, detalle_dir / "uso_system_prompt_por_idioma.png",
        color=P.C_VOTADOS, ylabel="%",
        value_fmt="%.1f", reference=100.0,
    )
    outputs.append(_emit_detail(path, "uso_system_prompt_por_idioma",
        f"Porcentaje de prompts que llevan un \\texttt{{system\\_prompt}} "
        f"no vacío, separado por idioma. Indicador de adopción del campo "
        f"opcional de preámbulo en la interfaz."
    ))

    # --- top 20 autores ---
    real = flat[~flat["is_v0_import"]]
    counts = real["username"].value_counts().head(20)
    path = P.detail_horizontal_bar(
        counts, detalle_dir / "top_autores.png",
        color=P.C_ENVIADOS, xlabel="prompts escritos",
    )
    outputs.append(_emit_detail(path, "top_autores",
        f"Los 20 autores más prolíficos del hackathon (excluyendo el "
        f"sentinel \\texttt{{v0}})."
    ))

    # --- histograma prompts por autor ---
    per_author = real["username"].value_counts()
    path = P.detail_histogram(
        per_author.values, detalle_dir / "histograma_prompts_por_autor.png",
        bins=25, color=P.C_VALIDADOS, xlabel="prompts por autor",
    )
    outputs.append(_emit_detail(path, "histograma_prompts_por_autor",
        f"Distribución del número de prompts escritos por cada autor "
        f"real ($n={len(per_author)}$ autores únicos)."
    ))

    # --- curva de Lorenz (standalone) ---
    g = metrics.gini(per_author.values) if not per_author.empty else 0.0
    path = P.detail_lorenz(
        per_author.values, detalle_dir / "curva_lorenz.png",
        gini=g, color=P.C_ENVIADOS,
    )
    outputs.append(_emit_detail(path, "curva_lorenz",
        f"Curva de Lorenz de la contribución de prompts por autor. El "
        f"coeficiente de Gini = ${g:.2f}$ (0 = perfecta equidad, "
        f"1 = un único autor escribe todo) cuantifica la desigualdad. "
        f"Métrica importante para citar el dataset: un Gini alto "
        f"significa que los datos reflejan la idiosincrasia de pocos "
        f"autores tanto como la del país que dicen representar."
    ))

    # --- distribución del estado de validación (standalone) ---
    state_counts = flat["validation_status"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    state_counts.index = ["0 slots", "1 slot", "2 slots", "3 slots\n(totalmente validado)"]
    path = P.detail_vertical_bar(
        state_counts, detalle_dir / "estado_validacion_por_prompt.png",
        color=P.C_VALIDADOS, ylabel="prompts",
    )
    outputs.append(_emit_detail(path, "estado_validacion_por_prompt",
        f"Número de prompts según el número de slots de validación "
        f"ocupados, sobre $n={n}$ prompts totales (incluye los $v0$ que "
        f"todavía no han pasado por validación humana)."
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
    summary = build_summary(flat)

    csv_path = io_utils.save_csv(summary, out_dir / "prompts.csv")
    n = len(flat)
    v0 = int(flat["is_v0_import"].sum()) if not flat.empty else 0
    table_caption = (
        f"Resumen de los $n={n}$ prompts escritos, agrupados por país. "
        f"Incluye prompts importados desde el dataset 2025 mediante el "
        f"sentinel \\texttt{{v0}} (en total {v0} prompts), que entran al "
        f"pipeline ya con par de respuestas pero sin pasar por validación "
        f"humana. La columna \\textit{{\\% con system prompt}} mide la "
        f"adopción del campo opcional que precede al prompt; la columna "
        f"\\textit{{\\% totalmente validados}} indica qué proporción ya "
        f"acumuló tres etiquetas de aceptación, condición para entrar a "
        f"la fase de votación."
    )
    latex_utils.save_table_tex(csv_path, summary, table_caption, label="prompts")
    print(f"wrote {csv_path}  ({len(summary)} filas)")

    panorama_path = plot_panorama_prompts(flat, plots_dir)
    real_authors = flat[~flat["is_v0_import"]] if not flat.empty else flat
    n_authors = real_authors["username"].nunique() if not real_authors.empty else 0
    g = metrics.gini(real_authors["username"].value_counts().values) if not real_authors.empty else 0.0
    latex_utils.save_figure_tex(
        panorama_path,
        caption=(
            f"Panorama de la escritura de prompts ($n={n}$ prompts, "
            f"$n={n_authors}$ autores reales sin contar el sentinel v0). "
            f"Panel superior izquierdo: embudo de prompts a lo largo de su "
            f"ciclo de vida — escritos, totalmente validados (los tres "
            f"validadores eligieron un bucket de aceptación), con par de "
            f"respuestas listo y totalmente votados. Panel superior derecho: "
            f"curva de Lorenz de la distribución de prompts por autor "
            f"(coeficiente de Gini = {g:.2f}; 0 sería equidad perfecta, 1 "
            f"sería un único autor escribiendo todo). Panel inferior "
            f"izquierdo: violinplot de la longitud (en caracteres) de cada "
            f"prompt, segmentado por idioma de la interfaz. Panel inferior "
            f"derecho: los 15 autores más prolíficos."
        ),
        label="panorama_prompts",
    )
    print(f"wrote {panorama_path}")

    mapa_path = plot_mapa_pais_idioma_estado(flat, plots_dir)
    latex_utils.save_figure_tex(
        mapa_path,
        caption=(
            f"Estado del pipeline por país. Panel izquierdo: número absoluto "
            f"de prompts por país, segmentado por idioma de la interfaz; "
            f"el color saturado señala los países con más volumen. Panel "
            f"derecho: distribución porcentual del estado de validación "
            f"dentro de cada país (qué porcentaje tiene 0, 1, 2 o 3 "
            f"validaciones registradas); el color saturado marca dónde se "
            f"acumulan prompts sin tocar y dónde se completan los tres "
            f"slots. Combinados, ambos paneles muestran al organizador "
            f"dónde tiene volumen pero falta validación, y dónde le falta "
            f"volumen sin más."
        ),
        label="mapa_pais_idioma_estado",
    )
    print(f"wrote {mapa_path}")

    detail_paths = emit_detail_plots(flat, detalle_dir)
    for p in detail_paths:
        print(f"wrote {p}")

    return [csv_path, panorama_path, mapa_path, *detail_paths]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
