"""Análisis de la votación de respuestas — sección 04.

  1. `panorama_votacion.png` (HEADLINE) — composición 2×2.
  2. `mapa_modelo_categoria.png` (HEADLINE) — heatmap modelo × categoría.
  3. `votacion.csv` + `votacion.tex` — resumen por modelo.
  4. `plots/detalle/*.png` + `.tex` — embudo de votación, distribución
     de elecciones (a/b/ambas/ninguna), tasas de victoria con IC95,
     tasa de indecisos por modelo, votos por categoría, votos por país,
     top votantes. Cobertura completa de las columnas
     `answer_chosen_{1,2,3}` y de los modelos en `model_{a,b}`.
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


VOTE_CHOICES = ("a", "b", "both", "none")


def _model_table(long: pd.DataFrame) -> pd.DataFrame:
    """Por modelo: apariciones, victorias, both, none. Apariciones = #votos
    en los que el modelo estaba en el par (regardless of slot)."""
    if long.empty:
        return pd.DataFrame(columns=["model", "appearances", "wins", "both", "none"])
    a_app = long.groupby("model_a").size().rename("a")
    b_app = long.groupby("model_b").size().rename("b")
    appearances = a_app.add(b_app, fill_value=0).rename("appearances")
    appearances = appearances[appearances.index.astype(str).str.strip() != ""]
    wins_a = long[long["choice"] == "a"].groupby("model_a").size().rename("wa")
    wins_b = long[long["choice"] == "b"].groupby("model_b").size().rename("wb")
    wins = wins_a.add(wins_b, fill_value=0).rename("wins")
    both_a = long[long["choice"] == "both"].groupby("model_a").size().rename("ba")
    both_b = long[long["choice"] == "both"].groupby("model_b").size().rename("bb")
    both = both_a.add(both_b, fill_value=0).rename("both")
    none_a = long[long["choice"] == "none"].groupby("model_a").size().rename("na")
    none_b = long[long["choice"] == "none"].groupby("model_b").size().rename("nb")
    none = none_a.add(none_b, fill_value=0).rename("none")
    table = (
        pd.concat([appearances, wins, both, none], axis=1)
        .fillna(0).astype(int).reset_index().rename(columns={"index": "model"})
    )
    return table


# ----- plot 1: panorama de votación ----------------------------------------


def plot_panorama_votacion(flat: pd.DataFrame, long: pd.DataFrame, out: Path) -> Path:
    fig = plt.figure(figsize=(16.0, 12.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.40)

    # --- Panel A: embudo de votación ---
    ax_a = fig.add_subplot(gs[0, 0])
    if flat.empty:
        P.empty(ax_a)
    else:
        has_ans = flat[flat["has_answers"]]
        counts = [
            len(has_ans),
            int((has_ans["n_votes"] >= 1).sum()),
            int((has_ans["n_votes"] >= 2).sum()),
            int((has_ans["n_votes"] >= 3).sum()),
        ]
        labels = ["con respuestas\n(a+b)", "1+ votos", "2+ votos", "3 votos\n(totalmente votado)"]
        P.funnel(ax_a, labels, counts, color=P.C_VOTADOS, base=len(has_ans) if len(has_ans) else 1)

    # --- Panel B: donut de elección ---
    ax_b = fig.add_subplot(gs[0, 1])
    if long.empty:
        P.empty(ax_b, "(aún no hay votos)")
    else:
        counts = long["choice"].value_counts().reindex(VOTE_CHOICES, fill_value=0)
        counts = counts[counts > 0]
        if counts.empty:
            P.empty(ax_b, "(sin votos válidos)")
        else:
            counts.index = counts.index.map(P.VOTE_LABELS_ES)
            palette_es = {P.VOTE_LABELS_ES[k]: v for k, v in P.VOTE_COLORS.items()}
            P.donut(ax_b, counts, palette=palette_es, hole_size=0.50,
                    legend_loc="center left", legend_bbox=(1.0, 0.5))

    # --- Panel C: forest plot — tasas de victoria con IC95 ---
    ax_c = fig.add_subplot(gs[1, 0])
    table = _model_table(long)
    if table.empty:
        P.empty(ax_c, "(aún no hay votos)")
    else:
        table["decisive"] = table["appearances"] - table["both"] - table["none"]
        eligible = table[table["decisive"] >= 5].copy()
        if eligible.empty:
            P.empty(ax_c, "(ningún modelo tiene n_decisive >= 5)")
        else:
            eligible["rate"] = eligible["wins"] / eligible["decisive"]
            ci = eligible.apply(lambda r: metrics.wilson_ci(int(r["wins"]), int(r["decisive"])), axis=1)
            eligible["lo"] = ci.apply(lambda t: t[0])
            eligible["hi"] = ci.apply(lambda t: t[1])
            P.errorbar_forest(
                ax_c, eligible,
                label_col="model", value_col="rate",
                lo_col="lo", hi_col="hi", n_col="decisive",
                color=P.C_VOTADOS, reference=0.5,
                xlabel="tasa de victoria (sin contar 'ambas' ni 'ninguna')",
            )

    # --- Panel D: tasa "ambas+ninguna" por modelo ---
    ax_d = fig.add_subplot(gs[1, 1])
    if table.empty:
        P.empty(ax_d, "(aún no hay votos)")
    else:
        eligible = table[table["appearances"] >= 5].copy()
        if eligible.empty:
            P.empty(ax_d, "(ningún modelo con apariciones >= 5)")
        else:
            eligible["pct_indecisive"] = (
                (eligible["both"] + eligible["none"]) / eligible["appearances"] * 100
            ).round(1)
            eligible = eligible.sort_values("pct_indecisive", ascending=False)
            eligible["label"] = eligible.apply(
                lambda r: f"{r['model']}\n(n={int(r['appearances'])})", axis=1
            )
            sns.barplot(
                data=eligible, y="label", x="pct_indecisive",
                color=P.C_PENDIENTE, ax=ax_d, edgecolor="white", linewidth=0.4,
            )
            for container in ax_d.containers:
                ax_d.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
            ax_d.set_xlabel("% votos 'ambas' + 'ninguna'")
            ax_d.set_ylabel("")

    return P.save_figure(fig, out / "panorama_votacion.png")


# ----- plot 2: mapa modelo × categoría -------------------------------------


def plot_mapa_modelo_categoria(long: pd.DataFrame, out: Path) -> Path:
    """Por cada (modelo, bucket de validación) muestra el número de
    victorias del modelo. Permite detectar especializaciones — ¿gana el
    modelo X en preguntas de 'conocimiento' pero pierde en 'trampa de
    sesgo'?"""
    out_path = out / "mapa_modelo_categoria.png"
    fig, ax = plt.subplots(figsize=(11.0, 7.0), facecolor="white")
    if long.empty:
        P.empty(ax, "(aún no hay votos)")
        return P.save_figure(fig, out_path)

    sub = long[long["primary_validation_bucket"] != ""].copy()
    if sub.empty:
        P.empty(ax, "(aún no hay votos sobre prompts validados)")
        return P.save_figure(fig, out_path)

    sub["chosen_clean"] = sub["chosen_model"].where(
        ~sub["choice"].isin(["both", "none"]), other=sub["choice"].map({"both": "ambas", "none": "ninguna"})
    )
    pivot = (
        sub.groupby(["chosen_clean", "primary_validation_bucket"]).size()
        .unstack(fill_value=0)
        .reindex(columns=list(_data.ACCEPT_VALIDATION_CHOICES), fill_value=0)
    )
    pivot.columns = [P.BUCKET_LABELS_ES[c] for c in pivot.columns]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot.index.name = None

    sns.heatmap(
        pivot, annot=True, fmt="d", cmap="rocket_r",
        linewidths=0.6, linecolor="white", ax=ax,
        cbar_kws={"label": "votos"},
        annot_kws={"fontsize": 8},
    )
    ax.set_xlabel("bucket de validación (dimensión cultural)")
    ax.set_ylabel("modelo elegido")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    return P.save_figure(fig, out_path)


# ----- summary table -------------------------------------------------------


def build_summary(long: pd.DataFrame) -> pd.DataFrame:
    table = _model_table(long)
    if table.empty:
        return pd.DataFrame(columns=[
            "Modelo", "Apariciones", "Victorias", "Tasa de victoria",
            "IC95 inf.", "IC95 sup.", "% indecisos",
        ])
    rows: list[dict] = []
    for _, r in table.iterrows():
        appearances = int(r["appearances"])
        decisive = appearances - int(r["both"]) - int(r["none"])
        wins = int(r["wins"])
        rate = wins / decisive if decisive > 0 else float("nan")
        lo, hi = metrics.wilson_ci(wins, decisive)
        indec = (int(r["both"]) + int(r["none"])) / appearances * 100 if appearances else 0.0
        rows.append({
            "Modelo": r["model"],
            "Apariciones": appearances,
            "Victorias": wins,
            "Tasa de victoria": round(rate, 3) if not np.isnan(rate) else None,
            "IC95 inf.": round(lo, 3),
            "IC95 sup.": round(hi, 3),
            "% indecisos": round(indec, 1),
        })
    return pd.DataFrame(rows).sort_values("Apariciones", ascending=False).reset_index(drop=True)


# ----- detail plots ---------------------------------------------------------


def _emit_detail(fig_path: Path, label: str, caption: str) -> Path:
    latex_utils.save_figure_tex(fig_path, caption=caption, label=label)
    return fig_path


def emit_detail_plots(flat: pd.DataFrame, long: pd.DataFrame, detalle_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if flat.empty:
        return outputs

    # --- embudo de votación (standalone) ---
    has_ans = flat[flat["has_answers"]]
    counts_funnel = [
        len(has_ans),
        int((has_ans["n_votes"] >= 1).sum()),
        int((has_ans["n_votes"] >= 2).sum()),
        int((has_ans["n_votes"] >= 3).sum()),
    ]
    labels_funnel = ["con respuestas (a+b)", "1+ votos", "2+ votos", "3 votos\n(totalmente votado)"]
    path = P.detail_funnel(
        labels_funnel, counts_funnel,
        detalle_dir / "embudo_votacion.png",
        color=P.C_VOTADOS, base=len(has_ans) if len(has_ans) else 1,
    )
    outputs.append(_emit_detail(path, "embudo_votacion",
        f"Embudo de votación: de los $n={len(has_ans)}$ prompts con par "
        f"$(a,b)$ de respuestas listo, cuántos han recibido 1, 2 o 3 "
        f"votos."
    ))

    if long.empty:
        return outputs

    # --- distribución de elecciones ---
    counts = long["choice"].value_counts().reindex(VOTE_CHOICES, fill_value=0)
    counts.index = [P.VOTE_LABELS_ES[c] for c in counts.index]
    multi = {P.VOTE_LABELS_ES[k]: v for k, v in P.VOTE_COLORS.items()}
    path = P.detail_vertical_bar(
        counts, detalle_dir / "distribucion_elecciones.png",
        color=P.C_VOTADOS, ylabel="votos",
    )
    outputs.append(_emit_detail(path, "distribucion_elecciones",
        f"Distribución de las $n={len(long)}$ elecciones registradas "
        f"entre los cuatro valores posibles: A, B, ambas, ninguna. "
        f"La suma de \\emph{{ambas}} y \\emph{{ninguna}} es el "
        f"indicador de prompts no discriminantes."
    ))

    # --- tasas de victoria con IC95 (standalone) ---
    table = _model_table(long)
    table["decisive"] = table["appearances"] - table["both"] - table["none"]
    eligible = table[table["decisive"] >= 5].copy()
    if not eligible.empty:
        eligible["rate"] = eligible["wins"] / eligible["decisive"]
        ci = eligible.apply(lambda r: metrics.wilson_ci(int(r["wins"]), int(r["decisive"])), axis=1)
        eligible["lo"] = ci.apply(lambda t: t[0])
        eligible["hi"] = ci.apply(lambda t: t[1])
        path = P.detail_errorbar(
            eligible, detalle_dir / "tasa_victoria_modelos.png",
            label_col="model", value_col="rate",
            lo_col="lo", hi_col="hi", n_col="decisive",
            color=P.C_VOTADOS, reference=0.5,
            xlabel="tasa de victoria (sin 'ambas' ni 'ninguna')",
        )
        outputs.append(_emit_detail(path, "tasa_victoria_modelos",
            f"Forest plot con la tasa de victoria de cada modelo y su "
            f"intervalo de confianza de Wilson al 95\\%, sobre votos "
            f"decisivos (excluyendo 'ambas' y 'ninguna'). La línea "
            f"discontinua marca el 50\\%; modelos con IC enteramente por "
            f"encima ganan sistemáticamente, los que están enteramente "
            f"por debajo pierden."
        ))

    # --- tasa de indecisos por modelo ---
    elig_app = table[table["appearances"] >= 5].copy()
    if not elig_app.empty:
        elig_app["pct"] = ((elig_app["both"] + elig_app["none"]) / elig_app["appearances"] * 100).round(1)
        elig_app = elig_app.sort_values("pct", ascending=False)
        ser = pd.Series(
            elig_app["pct"].values,
            index=[f"{m} (n={int(a)})" for m, a in zip(elig_app["model"], elig_app["appearances"])],
        )
        path = P.detail_horizontal_bar(
            ser, detalle_dir / "tasa_indecisos_modelos.png",
            color=P.C_PENDIENTE, xlabel="% votos 'ambas' + 'ninguna'",
        )
        outputs.append(_emit_detail(path, "tasa_indecisos_modelos",
            f"Para cada modelo: porcentaje de sus apariciones (en cualquiera "
            f"de las dos posiciones del par) en las que el votante eligió "
            f"\\emph{{ambas}} o \\emph{{ninguna}}. Valores altos pueden "
            f"reflejar un modelo cuya respuesta es indistinguible de "
            f"sus oponentes, o un modelo que alucina con tal confianza "
            f"que el votante prefiere descartar las dos."
        ))

    # --- votos por categoría de validación (stacked) ---
    sub = long[long["primary_validation_bucket"] != ""]
    if not sub.empty:
        tab = (
            sub.groupby(["primary_validation_bucket", "choice"]).size()
            .unstack(fill_value=0)
            .reindex(columns=list(VOTE_CHOICES), fill_value=0)
        )
        tab.index = [P.BUCKET_LABELS_ES.get(c, c) for c in tab.index]
        tab.index.name = "categoria"
        tab.columns = [P.VOTE_LABELS_ES[c] for c in tab.columns]
        palette_vote_es = {P.VOTE_LABELS_ES[k]: v for k, v in P.VOTE_COLORS.items()}
        tab_wide = tab.reset_index()
        path = P.detail_stacked_bar(
            tab_wide, detalle_dir / "votos_por_categoria.png",
            x_col="categoria", stack_cols=list(tab.columns),
            palette=palette_vote_es, ylabel="votos",
            rotate_xticks=20,
        )
        outputs.append(_emit_detail(path, "votos_por_categoria",
            f"Composición de elecciones (A/B/ambas/ninguna) dentro de "
            f"cada categoría de validación dominante del prompt. "
            f"Revela patrones por dimensión cultural."
        ))

    # --- votos por país (stacked) ---
    tab_pais = (
        long.groupby(["prompt_country_display", "choice"]).size()
        .unstack(fill_value=0)
        .reindex(columns=list(VOTE_CHOICES), fill_value=0)
    )
    tab_pais = tab_pais.loc[tab_pais.sum(axis=1).sort_values(ascending=False).index]
    tab_pais.columns = [P.VOTE_LABELS_ES[c] for c in tab_pais.columns]
    palette_vote_es = {P.VOTE_LABELS_ES[k]: v for k, v in P.VOTE_COLORS.items()}
    tab_wide_pais = tab_pais.reset_index().rename(columns={"prompt_country_display": "pais"})
    path = P.detail_stacked_bar(
        tab_wide_pais, detalle_dir / "votos_por_pais.png",
        x_col="pais", stack_cols=list(tab_pais.columns),
        palette=palette_vote_es, ylabel="votos",
        orientation="horizontal",
    )
    outputs.append(_emit_detail(path, "votos_por_pais",
        f"Composición de elecciones por país de origen del prompt. "
        f"Diferencias significativas entre países en preferencia A/B "
        f"sugieren especialización cultural-regional de los modelos."
    ))

    # --- top votantes ---
    counts = long["voter_username"].value_counts().head(20)
    path = P.detail_horizontal_bar(
        counts, detalle_dir / "top_votantes.png",
        color=P.C_VOTADOS, xlabel="votos emitidos",
    )
    outputs.append(_emit_detail(path, "top_votantes",
        f"Los 20 votantes más activos del hackathon."
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
    long = dl.load_votes_long(prompts_df)
    summary = build_summary(long)

    csv_path = io_utils.save_csv(summary, out_dir / "votacion.csv")
    n_votes = len(long)
    n_voters = long["voter_username"].nunique() if not long.empty else 0
    table_caption = (
        f"Rendimiento de cada modelo en la votación de respuestas "
        f"($n={n_votes}$ votos emitidos por $n={n_voters}$ votantes "
        f"únicos sobre el corte de datos actual). La tasa de victoria se "
        f"calcula sobre votos \\textit{{decisivos}} (excluyendo los votos "
        f"\\texttt{{ambas}} y \\texttt{{ninguna}}) y se acompaña de un "
        f"intervalo de confianza de Wilson al 95\\% para evitar afirmaciones "
        f"basadas en muestras pequeñas. La columna \\textit{{\\% indecisos}} "
        f"mide la fracción de votos en que el votante no pudo distinguir el "
        f"modelo del oponente — una señal indirecta de calidad del prompt."
    )
    latex_utils.save_table_tex(csv_path, summary, table_caption, label="votacion")
    print(f"wrote {csv_path}  ({len(summary)} filas)")

    panorama_path = plot_panorama_votacion(flat, long, plots_dir)
    latex_utils.save_figure_tex(
        panorama_path,
        caption=(
            f"Panorama de la votación de respuestas. Panel superior "
            f"izquierdo: embudo del estado del pipeline de votación, "
            f"empezando por los prompts con par $(a,b)$ de respuestas y "
            f"avanzando a 1, 2 y 3 votos. Panel superior derecho: "
            f"distribución de las elecciones del votante "
            f"(A / B / ambas / ninguna). Panel inferior izquierdo: "
            f"\\textit{{forest plot}} con la tasa de victoria de cada "
            f"modelo y su intervalo de confianza de Wilson al 95\\%; la "
            f"línea discontinua marca el 50\\%. Panel inferior derecho: "
            f"porcentaje de votos en los que el votante eligió 'ambas' o "
            f"'ninguna' cuando el modelo formaba parte del par — una alta "
            f"tasa indica que el modelo tiende a quedar emparejado en "
            f"comparaciones poco discriminantes."
        ),
        label="panorama_votacion",
    )
    print(f"wrote {panorama_path}")

    mapa_path = plot_mapa_modelo_categoria(long, plots_dir)
    latex_utils.save_figure_tex(
        mapa_path,
        caption=(
            f"Modelos elegidos por dimensión cultural del prompt. Cada "
            f"celda muestra cuántas veces el modelo de la fila fue "
            f"elegido como ganador en prompts cuyo bucket de validación "
            f"mayoritario era el de la columna. Permite identificar "
            f"especializaciones — por ejemplo, un modelo que domina en "
            f"\\textit{{conocimiento}} pero queda relegado en "
            f"\\textit{{trampa de sesgo}}. Las filas \\texttt{{ambas}} y "
            f"\\texttt{{ninguna}} agregan los votos en los que el "
            f"comparador no logró distinguir las dos respuestas."
        ),
        label="mapa_modelo_categoria",
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
