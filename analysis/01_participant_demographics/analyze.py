"""Análisis demográfico — sección 01.

Produce:

  1. `panorama_demografico.png` (HEADLINE) — composición 2×2 densa.
  2. `mapa_pais_perfil.png` (HEADLINE) — heatmap país × dimensión.
  3. `participantes.csv` + `.tex` — tabla resumen por país.
  4. `plots/detalle/*.png` + `.tex` — un plot por dimensión demográfica
     (país, idioma, país×idioma, puntaje del test, tasa de aprobación,
     pronombres, educación, campo de estudio, nivel de PLN, primera vez,
     año de nacimiento). Cubren TODO lo que está en el dataset de
     participantes para que el informe pueda mostrar cualquier corte.
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
from analysis._common import io_utils, latex_utils, plotting as P

import matplotlib.pyplot as plt
import seaborn as sns


# ----- plot 1: panorama demográfico ----------------------------------------


def plot_panorama_demografico(participants: pd.DataFrame, out: Path) -> Path:
    """Composición 2×2 que cubre país, idioma, puntaje del test y dos
    dimensiones del perfil del participante (pronombres + nivel de NLP)."""
    fig = plt.figure(figsize=(13.5, 10.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # --- Panel A: países, hue=passed_test ---
    ax_a = fig.add_subplot(gs[0, 0])
    if participants.empty:
        P.empty(ax_a)
    else:
        country_order = (
            participants["country_display"]
            .value_counts()
            .index.tolist()
        )
        sns.countplot(
            data=participants,
            y="country_display",
            hue="passed_test",
            order=country_order,
            palette={True: P.C_APROBADO, False: P.C_NEUTRO},
            ax=ax_a, edgecolor="white", linewidth=0.4,
        )
        ax_a.set_xlabel("participantes")
        ax_a.set_ylabel("")
        leg = ax_a.legend(title="¿aprobó test?", fontsize=8, title_fontsize=8, frameon=False)
        for handle, label in zip(leg.legend_handles, ["no", "sí"]):
            handle.set_label(label)
        ax_a.legend(handles=leg.legend_handles, labels=["no", "sí"],
                    title="¿aprobó test?", fontsize=8, title_fontsize=8, frameon=False)

    # --- Panel B: puntaje del test, KDE por idioma ---
    ax_b = fig.add_subplot(gs[0, 1])
    scores = participants[["best_test_score", "language"]].dropna()
    scores = scores[scores["best_test_score"] > 0]  # KDE no funciona en una sola masa puntual
    if scores.empty:
        P.empty(ax_b, message="(nadie ha completado el test todavía)")
    else:
        scores = scores.assign(idioma=scores["language"].map(P.LANGUAGE_LABELS_ES).fillna(scores["language"]))
        palette = {P.LANGUAGE_LABELS_ES[k]: v for k, v in P.LANGUAGE_COLORS.items() if k in P.LANGUAGE_LABELS_ES}
        sns.kdeplot(
            data=scores, x="best_test_score", hue="idioma",
            palette=palette, ax=ax_b, fill=True,
            common_norm=False, alpha=0.45, linewidth=1.5, clip=(0, 16),
        )
        ax_b.axvline(_data.TEST_PASS_THRESHOLD, color="black", linestyle="--", linewidth=1.2)
        top = ax_b.get_ylim()[1]
        ax_b.text(_data.TEST_PASS_THRESHOLD, top * 0.95,
                  f"  umbral ({_data.TEST_PASS_THRESHOLD:g})",
                  ha="left", va="top", fontsize=9)
        ax_b.set_xlabel("puntaje (sobre 16)")
        ax_b.set_ylabel("densidad")
        ax_b.set_xlim(0, 16)

    # --- Panel C: donut de pronombres ---
    ax_c = fig.add_subplot(gs[1, 0])
    if "pronouns_norm" not in participants.columns:
        P.empty(ax_c)
    else:
        raw = (
            participants["pronouns_norm"].replace("", "(sin respuesta)").value_counts()
        )
        counts = _translate_index(raw, "pronouns_norm")
        palette_pn = {
            "él": "#3b82f6",
            "ella": "#ec4899",
            "elle / they": "#8b5cf6",
            "prefiere no decir": P.C_NEUTRO,
            "(sin respuesta)": "#e5e7eb",
        }
        P.donut(ax_c, counts, palette=palette_pn, legend_loc="center left",
                legend_bbox=(1.05, 0.5))

    # --- Panel D: barra polar de nivel de NLP × idioma ---
    ax_d = fig.add_subplot(gs[1, 1], projection="polar")
    if "nlp_level_norm" not in participants.columns or participants["nlp_level_norm"].astype(str).str.strip().eq("").all():
        ax_d.text(0, 0, "(sin datos)", ha="center", va="center")
        ax_d.set_axis_off()
    else:
        nlp_order = ["Basic", "Intermediate", "Advanced", "(no answer)"]
        counts = (
            participants["nlp_level_norm"]
            .replace("", "(no answer)")
            .value_counts()
            .reindex(nlp_order, fill_value=0)
        )
        labels_es = {
            "Basic": "básico", "Intermediate": "intermedio",
            "Advanced": "avanzado", "(no answer)": "sin respuesta",
        }
        P.radial_bar(
            ax_d,
            [labels_es[k] for k in counts.index],
            counts.values.tolist(),
            palette=["#a5b4fc", P.C_ENVIADOS, P.C_VALIDADOS, P.C_NEUTRO],
        )

    return P.save_figure(fig, out / "panorama_demografico.png")


# ----- plot 2: clustermap país × perfil ------------------------------------


def plot_mapa_pais_perfil(participants: pd.DataFrame, out: Path) -> Path:
    """Mapa de calor país × dimensión demográfica. Cada fila se normaliza
    para mostrar la composición proporcional dentro de ese país (no el
    número absoluto), de modo que países con pocos participantes son
    comparables con países grandes. Países con n<3 se excluyen."""
    out_path = out / "mapa_pais_perfil.png"
    fig, ax = plt.subplots(figsize=(11.0, 7.5), facecolor="white")
    if participants.empty:
        P.empty(ax)
        return P.save_figure(fig, out_path)

    df = participants.copy()
    # Sólo países con al menos 3 participantes — el resto produce ruido.
    size_per_country = df.groupby("country_display").size()
    keep = size_per_country[size_per_country >= 3].index.tolist()
    df = df[df["country_display"].isin(keep)]
    if df.empty:
        P.empty(ax, "(no hay países con n>=3 participantes)")
        return P.save_figure(fig, out_path)

    # Vector de perfil por país: % de cada categoría en varias dimensiones.
    def dim_share(col: str, value: str, label: str) -> pd.Series:
        return (
            df.assign(_hit=df[col].astype(str).str.contains(value, case=False, na=False))
            .groupby("country_display")["_hit"]
            .mean()
            .mul(100)
            .rename(label)
        )

    rows = pd.concat(
        [
            df.groupby("country_display")["passed_test"].mean().mul(100).rename("% aprobó test"),
            dim_share("nlp_level_norm", "Advanced", "% NLP avanzado"),
            dim_share("nlp_level_norm", "Intermediate", "% NLP intermedio"),
            dim_share("nlp_level_norm", "Basic", "% NLP básico"),
            dim_share("education_norm", "Doctoral", "% doctorado"),
            dim_share("education_norm", "Master", "% maestría"),
            dim_share("education_norm", "Bachelor", "% grado/licenciatura"),
            dim_share("first_event_norm", "First", "% primera vez"),
            dim_share("field_norm", "Computer", "% comp. sci."),
            dim_share("field_norm", "Engineering", "% ingeniería"),
            dim_share("field_norm", "Humanities", "% artes y humanidades"),
        ],
        axis=1,
    ).fillna(0).round(1)

    # Reordenar países por aprobación del test (descendente) para que la
    # estructura del mapa salte a la vista.
    rows = rows.sort_values("% aprobó test", ascending=False)

    sns.heatmap(
        rows, annot=True, fmt=".0f", cmap="rocket_r",
        linewidths=0.6, linecolor="white", ax=ax,
        cbar_kws={"label": "% de participantes en el país"},
        annot_kws={"fontsize": 8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    return P.save_figure(fig, out_path)


# ----- summary table -------------------------------------------------------


def build_summary(participants: pd.DataFrame) -> pd.DataFrame:
    if participants.empty:
        return pd.DataFrame(columns=[
            "País", "n", "% aprobó test", "Puntaje medio",
            "Idioma dominante", "NLP avanzado (%)",
        ])
    grouped = participants.groupby("country_display", dropna=False)
    rows: list[dict] = []
    for country, sub in grouped:
        n = len(sub)
        passed = float(sub["passed_test"].mean() * 100)
        mean_score = float(sub["best_test_score"].mean())
        dom_lang_code = sub["language"].mode().iloc[0] if not sub["language"].mode().empty else ""
        dom_lang = P.LANGUAGE_LABELS_ES.get(dom_lang_code, dom_lang_code or "—")
        adv_pct = float(
            sub["nlp_level_norm"].astype(str).str.contains("Advanced", case=False, na=False).mean() * 100
        ) if "nlp_level_norm" in sub.columns else 0.0
        rows.append({
            "País": country if country else "(sin dato)",
            "n": n,
            "% aprobó test": round(passed, 1),
            "Puntaje medio": round(mean_score, 2),
            "Idioma dominante": dom_lang,
            "NLP avanzado (%)": round(adv_pct, 1),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ----- detail plots ---------------------------------------------------------
# Un PNG + un .tex por cada dimensión. Cobertura completa del dataset.


# Spanish display labels for the normalized values returned by data_loading.
# The normalizers keep canonical English keys for stability; the translation
# applies only to what's printed on the plot.
_TRANSLATIONS = {
    "pronouns_norm": {
        "he / él": "él",
        "she / ella": "ella",
        "they / elle": "elle / they",
        "prefer not to say": "prefiere no decir",
    },
    "education_norm": {
        "Tertiary — Bachelor's": "Grado / Licenciatura",
        "Tertiary — Master's": "Máster",
        "Tertiary — Doctoral": "Doctorado",
        "Tertiary — short-cycle": "Técnico superior",
        "Post-secondary": "Postsecundario",
        "Upper secondary": "Bachillerato",
        "Lower secondary": "Secundaria",
    },
    "field_norm": {
        "Computer science": "Informática",
        "Engineering": "Ingeniería",
        "Natural sci. & maths": "Ciencias naturales y matemáticas",
        "Arts & humanities": "Artes y humanidades",
        "Social sci. & journalism": "Ciencias sociales y periodismo",
        "Education": "Educación",
        "Business & law": "Empresariales y derecho",
        "Services": "Servicios",
        "Agriculture & veterinary": "Agricultura y veterinaria",
        "Health & welfare": "Salud y bienestar",
        "Other": "Otros",
    },
    "nlp_level_norm": {
        "Basic": "Básico",
        "Intermediate": "Intermedio",
        "Advanced": "Avanzado",
    },
    "first_event_norm": {
        "First SomosNLP event": "Primera vez",
        "Returning attendee": "Asistente recurrente",
    },
    "birth_year_bucket": {
        "before 1990": "antes de 1990",
        "2005 or later": "2005 o posterior",
    },
}
_TRANSLATIONS_COMMON = {"(no answer)": "(sin respuesta)", "(sin respuesta)": "(sin respuesta)", "": "(sin respuesta)"}


def _translate_index(counts: pd.Series, col: str) -> pd.Series:
    mapping = {**_TRANSLATIONS_COMMON, **_TRANSLATIONS.get(col, {})}
    new_index = [mapping.get(idx, idx) for idx in counts.index]
    # If duplicates appear after translation (shouldn't, but just in case),
    # collapse by summing.
    return pd.Series(counts.values, index=new_index).groupby(level=0).sum().sort_values(ascending=False)


_DETAIL_DEMOGRAPHIC_SPECS = [
    # (column, plot_filename, label_es, color, multi)
    ("pronouns_norm",    "pronombres",        "pronombres",                P.C_ENVIADOS, False),
    ("education_norm",   "educacion",         "nivel educativo",           P.C_VALIDADOS, False),
    ("field_norm",       "campo_estudio",     "campo de estudio o trabajo", P.C_VOTADOS, True),
    ("nlp_level_norm",   "nivel_nlp",         "nivel de PLN autodeclarado", P.C_HIGHLIGHT, False),
    ("first_event_norm", "primera_vez_somosnlp", "¿primera vez en SomosNLP?", P.C_APROBADO, False),
    ("birth_year_bucket", "anio_nacimiento",   "año de nacimiento (bucketed)", P.C_NEUTRO, False),
]


def _exploded_counts(participants: pd.DataFrame, col: str, multi: bool) -> pd.Series:
    if col not in participants.columns or participants[col].astype(str).str.strip().eq("").all():
        return pd.Series(dtype=int)
    if multi:
        bucket: list[str] = []
        for cell in participants[col].dropna():
            parts = [p.strip() for p in str(cell).split("|") if p.strip()]
            bucket.extend(parts)
        return pd.Series(bucket).value_counts() if bucket else pd.Series(dtype=int)
    return participants[col].replace("", "(sin respuesta)").value_counts()


def _emit_detail(
    fig_path: Path,
    label: str,
    caption: str,
) -> Path:
    latex_utils.save_figure_tex(fig_path, caption=caption, label=label)
    return fig_path


def emit_detail_plots(participants: pd.DataFrame, detalle_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    n = len(participants)
    if participants.empty:
        return outputs

    # --- por país ---
    counts = participants["country_display"].fillna("?").value_counts()
    path = P.detail_horizontal_bar(
        counts, detalle_dir / "participantes_por_pais.png",
        color=P.C_ENVIADOS, xlabel="participantes",
    )
    outputs.append(_emit_detail(path, "participantes_por_pais",
        f"Distribución de los {n} participantes registrados por país, "
        f"derivada del campo \\texttt{{country}} del dataset "
        f"\\texttt{{hackathon\\_participants}} mediante el mapeo de "
        f"\\texttt{{COUNTRY\\_PATTERNS}} a códigos ISO. Países sin código "
        f"reconocido aparecen como `?`."
    ))

    # --- por idioma ---
    counts = (
        participants["language"].fillna("?")
        .map(P.LANGUAGE_LABELS_ES).fillna(participants["language"].fillna("?"))
        .value_counts()
    )
    path = P.detail_vertical_bar(
        counts, detalle_dir / "participantes_por_idioma.png",
        color=P.C_VALIDADOS, ylabel="participantes",
    )
    outputs.append(_emit_detail(path, "participantes_por_idioma",
        f"Distribución de los {n} participantes por idioma de la "
        f"interfaz elegido al registrarse en Eventbrite (ticket "
        f"\\texttt{{Hackathon (Español/English/Portugues)}})."
    ))

    # --- país × idioma (stacked) ---
    tab = (
        participants.groupby(["country_display", "language"]).size()
        .unstack(fill_value=0)
    )
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    tab.columns = [P.LANGUAGE_LABELS_ES.get(c, c) for c in tab.columns]
    palette_lang = {P.LANGUAGE_LABELS_ES[k]: v for k, v in P.LANGUAGE_COLORS.items()}
    tab_wide = tab.reset_index()
    path = P.detail_stacked_bar(
        tab_wide, detalle_dir / "participantes_pais_x_idioma.png",
        x_col="country_display", stack_cols=list(tab.columns),
        palette=palette_lang, ylabel="participantes",
        orientation="horizontal",
    )
    outputs.append(_emit_detail(path, "participantes_pais_x_idioma",
        f"Composición lingüística por país. Permite detectar desajustes "
        f"entre el país declarado y el idioma de la interfaz utilizado "
        f"durante el registro — por ejemplo, participantes brasileños "
        f"usando la interfaz en español o inglés en vez de la "
        f"traducción al portugués."
    ))

    # --- distribución del puntaje del test (histograma) ---
    path = P.detail_histogram(
        participants["best_test_score"], detalle_dir / "puntaje_test_distribucion.png",
        bins=17, color=P.C_HIGHLIGHT,
        xlabel="mejor puntaje (sobre 16)",
        vline=_data.TEST_PASS_THRESHOLD,
        vline_label=f"umbral ({_data.TEST_PASS_THRESHOLD:g})",
    )
    outputs.append(_emit_detail(path, "puntaje_test_distribucion",
        f"Histograma del mejor puntaje alcanzado por cada participante "
        f"en el test de entrada, considerando todos los intentos. La "
        f"línea discontinua marca el umbral de aprobación "
        f"(\\texttt{{TEST\\_PASS\\_THRESHOLD = {_data.TEST_PASS_THRESHOLD:g}}})."
    ))

    # --- tasa de aprobación por idioma ---
    rate = (
        participants.groupby("language")["passed_test"]
        .mean().mul(100).round(1).sort_index()
    )
    counts_lang = participants.groupby("language").size()
    rate.index = [f"{P.LANGUAGE_LABELS_ES.get(k, k)} (n={int(counts_lang[k])})" for k in rate.index]
    path = P.detail_vertical_bar(
        rate, detalle_dir / "tasa_aprobacion_por_idioma.png",
        color=P.C_APROBADO, ylabel="% participantes",
        value_fmt="%.1f", reference=100.0,
    )
    outputs.append(_emit_detail(path, "tasa_aprobacion_por_idioma",
        f"Porcentaje de participantes que aprobaron el test "
        f"(umbral \\texttt{{best\\_test\\_score $\\geq$ "
        f"{_data.TEST_PASS_THRESHOLD:g}}}), separado por idioma de la "
        f"interfaz. Diferencias grandes entre idiomas indicarían que el "
        f"banco de preguntas traducido tiene problemas de calibración."
    ))

    # --- cada dimensión demográfica ---
    for col, fname, label, color, multi in _DETAIL_DEMOGRAPHIC_SPECS:
        counts = _translate_index(_exploded_counts(participants, col, multi), col)
        path = P.detail_horizontal_bar(
            counts, detalle_dir / f"{fname}_distribucion.png",
            color=color, xlabel="participantes",
        )
        suffix = " Pregunta multi-respuesta (cada participante puede elegir varias)." if multi else ""
        outputs.append(_emit_detail(path, f"{fname}_distribucion",
            f"Distribución del cohorte por {label}, normalizada al "
            f"conjunto canónico de valores resuelto en "
            f"\\texttt{{analysis/\\_common/data\\_loading.norm\\_*}}."
            f"{suffix}"
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

    enriched = dl.load_participants_enriched(participants_df, prompts_df)
    summary = build_summary(enriched)

    csv_path = io_utils.save_csv(summary, out_dir / "participantes.csv")
    n = len(enriched)
    passed = int(enriched["passed_test"].sum()) if not enriched.empty else 0
    table_caption = (
        f"Composición del cohorte de participantes ($n={n}$). "
        f"Cada fila resume los participantes registrados de un país, "
        f"incluyendo el porcentaje que aprobó el test de entrada "
        f"(umbral = {_data.TEST_PASS_THRESHOLD:g} puntos sobre 16), "
        f"su puntaje medio, el idioma de la interfaz mayoritario y "
        f"la proporción que se autodeclaró con nivel avanzado de PLN. "
        f"Los datos demográficos provienen del registro de Eventbrite, "
        f"unidos por usuario de Hugging Face."
    )
    latex_utils.save_table_tex(csv_path, summary, table_caption, label="participantes")
    print(f"wrote {csv_path}  ({len(summary)} filas)")

    panorama_path = plot_panorama_demografico(enriched, plots_dir)
    latex_utils.save_figure_tex(
        panorama_path,
        caption=(
            f"Panorama demográfico de los {n} participantes registrados. "
            f"Panel superior izquierdo: distribución por país, con barras "
            f"coloreadas según si el participante aprobó el test de entrada "
            f"(verde = sí, gris = no). Panel superior derecho: estimación "
            f"de densidad de los puntajes del test, segmentada por idioma "
            f"de la interfaz; la línea discontinua marca el umbral de "
            f"aprobación de {_data.TEST_PASS_THRESHOLD:g} sobre 16. "
            f"Panel inferior izquierdo: distribución de pronombres como gráfico "
            f"de dona. Panel inferior derecho: barras polares con el nivel "
            f"autodeclarado de conocimiento de PLN."
        ),
        label="panorama_demografico",
    )
    print(f"wrote {panorama_path}")

    mapa_path = plot_mapa_pais_perfil(enriched, plots_dir)
    latex_utils.save_figure_tex(
        mapa_path,
        caption=(
            f"Perfil demográfico por país (sólo países con $n \\geq 3$ "
            f"participantes). Cada celda muestra el porcentaje de "
            f"participantes del país (fila) que pertenece a la categoría "
            f"correspondiente (columna). Las filas están ordenadas por "
            f"porcentaje de aprobación del test, de mayor a menor, para "
            f"que se distinga visualmente la relación entre cohortes con "
            f"mejor rendimiento y su composición educativa y de "
            f"experiencia previa en PLN."
        ),
        label="mapa_pais_perfil",
    )
    print(f"wrote {mapa_path}")

    detail_paths = emit_detail_plots(enriched, detalle_dir)
    for p in detail_paths:
        print(f"wrote {p}")

    return [csv_path, panorama_path, mapa_path, *detail_paths]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
