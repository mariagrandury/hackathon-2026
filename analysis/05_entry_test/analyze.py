"""Análisis del test de entrada — sección 05.

  1. `matriz_confusion.png` (HEADLINE) — matriz de confusión + accuracy
     por categoría.
  2. `dificultad_preguntas.png` (HEADLINE) — accuracy por pregunta.
  3. `test.csv` + `test.tex` — resumen por categoría.
  4. `plots/detalle/*.png` + `.tex` — accuracy por categoría
     (standalone), progresión por intento, tasa de aprobación por
     idioma, distribución de intentos por usuario, accuracy MCQ (si
     hay suficientes). Cobertura completa de las columnas
     `test_score` y `test_responses`.
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
import test_data as _test_data
from analysis._common import data_loading as dl
from analysis._common import io_utils, latex_utils, plotting as P

import matplotlib.pyplot as plt
import seaborn as sns


# ----- plot 1: matriz de confusión + accuracy por categoría ---------------


def plot_matriz_confusion(long: pd.DataFrame, out: Path) -> Path:
    """Composición 1×1 dominante (matriz) con barra de accuracy por
    categoría apilada debajo, en proporción 4:1."""
    out_path = out / "matriz_confusion.png"
    fig = plt.figure(figsize=(11.5, 12.0), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.30, height_ratios=[4.0, 1.5])

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    sub = long[(long["format"] == "classification") & long["correct_key"].astype(bool)]
    if sub.empty:
        P.empty(ax_top)
        P.empty(ax_bot)
        return P.save_figure(fig, out_path)

    cats = list(_data.VALIDATION_CHOICES)
    es_cats = [P.BUCKET_LABELS_ES[c] for c in cats]
    pivot = (
        sub.pivot_table(
            index="correct_key", columns="chosen",
            values="question_id", aggfunc="size", fill_value=0,
        )
        .reindex(index=cats, columns=cats, fill_value=0)
    )
    row_totals = pivot.sum(axis=1).replace(0, 1)
    norm = pivot.div(row_totals, axis=0)
    norm.index = es_cats
    norm.columns = es_cats

    sns.heatmap(
        norm, annot=True, fmt=".0%", cmap="Blues",
        linewidths=0.6, linecolor="white", ax=ax_top,
        cbar_kws={"label": "% del bucket correcto"},
        annot_kws={"fontsize": 9}, vmin=0, vmax=1,
    )
    ax_top.set_xlabel("bucket elegido por el participante")
    ax_top.set_ylabel("bucket correcto")
    for tick in ax_top.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")

    # Bottom: barra de accuracy estricta por categoría
    by_cat = (
        sub.groupby("category")
        .agg(n=("is_correct_strict", "size"), correct=("is_correct_strict", "sum"))
    )
    by_cat["acc"] = (by_cat["correct"] / by_cat["n"] * 100).round(1)
    by_cat = by_cat.reindex(cats, fill_value=0)
    labels = [f"{P.BUCKET_LABELS_ES[c]}\n(n={int(by_cat.loc[c, 'n'])})" for c in cats]
    palette = [P.BUCKET_COLORS[c] for c in cats]
    sns.barplot(
        x=labels, y=by_cat["acc"].values, hue=labels,
        palette=palette, ax=ax_bot, edgecolor="white", linewidth=0.4,
        legend=False,
    )
    for container in ax_bot.containers:
        ax_bot.bar_label(container, fmt="%.0f%%", fontsize=9, padding=2)
    ax_bot.set_xlabel("")
    ax_bot.set_ylabel("accuracy estricta (%)")
    ax_bot.set_ylim(0, 110)
    ax_bot.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.5)

    return P.save_figure(fig, out_path)


# ----- plot 2: dificultad por pregunta -------------------------------------


def plot_dificultad_preguntas(long: pd.DataFrame, out: Path) -> Path:
    """Una barra por pregunta de clasificación, ordenadas por accuracy
    ascendente. Color = categoría correcta. Permite ver de un vistazo
    qué items son candidatos a rotar."""
    out_path = out / "dificultad_preguntas.png"
    sub = long[long["format"] == "classification"]
    if sub.empty:
        fig, ax = plt.subplots(figsize=(11.5, 8.0), facecolor="white")
        P.empty(ax)
        return P.save_figure(fig, out_path)
    grouped = (
        sub.groupby(["question_id", "category"], dropna=False)
        .agg(
            n=("is_correct_strict", "size"),
            correct=("is_correct_strict", "sum"),
        )
        .reset_index()
    )
    grouped["acc"] = (grouped["correct"] / grouped["n"] * 100).round(1)
    grouped = grouped.sort_values("acc").reset_index(drop=True)
    grouped["categoria_es"] = grouped["category"].map(P.BUCKET_LABELS_ES).fillna(grouped["category"])
    grouped["label"] = grouped.apply(lambda r: f"{r['question_id']}  ({r['categoria_es']}, n={int(r['n'])})", axis=1)

    height = max(8.0, 0.30 * len(grouped) + 2.0)
    fig, ax = plt.subplots(figsize=(13.5, height), facecolor="white")
    palette_lookup = {P.BUCKET_LABELS_ES[k]: v for k, v in P.BUCKET_COLORS.items()}
    palette_lookup.update({k: v for k, v in P.BUCKET_COLORS.items()})
    sns.barplot(
        data=grouped, y="label", x="acc", hue="categoria_es",
        palette=palette_lookup,
        ax=ax, edgecolor="white", linewidth=0.3, dodge=False,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", fontsize=8, padding=2)
    ax.set_xlabel("accuracy estricta (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 110)
    ax.legend(title="categoría correcta", fontsize=8, title_fontsize=8,
              frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    return P.save_figure(fig, out_path)


# ----- summary table -------------------------------------------------------


def build_summary(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=[
            "Categoría", "Lado", "n respuestas",
            "Accuracy estricta (%)", "% mismo lado", "Puntaje medio",
        ])
    sub = long[long["format"] == "classification"]
    rows = []
    for cat in _data.VALIDATION_CHOICES:
        cat_sub = sub[sub["category"] == cat]
        n = len(cat_sub)
        if n == 0:
            continue
        side = "aceptación" if cat in _data.ACCEPT_VALIDATION_CHOICES else "rechazo"
        strict = float(cat_sub["is_correct_strict"].mean() * 100)
        # "mismo lado" = chose a category on the same accept/reject side
        chosen_side = cat_sub["chosen"].apply(
            lambda c: "accept" if c in _data.ACCEPT_VALIDATION_CHOICES
            else ("reject" if c in _data.REJECT_CHOICES else "")
        )
        correct_side = "accept" if side == "aceptación" else "reject"
        same_side_rate = float((chosen_side == correct_side).mean() * 100)
        rows.append({
            "Categoría": P.BUCKET_LABELS_ES[cat],
            "Lado": side,
            "n respuestas": n,
            "Accuracy estricta (%)": round(strict, 1),
            "% mismo lado": round(same_side_rate, 1),
            "Puntaje medio": round(float(cat_sub["points_earned"].mean()), 2),
        })
    return pd.DataFrame(rows)


# ----- detail plots ---------------------------------------------------------

MCQ_PLOT_MIN_ITEMS = 3  # only emit the MCQ plot if the bank has >= 3 MCQs


def _emit_detail(fig_path: Path, label: str, caption: str) -> Path:
    latex_utils.save_figure_tex(fig_path, caption=caption, label=label)
    return fig_path


def emit_detail_plots(long: pd.DataFrame, participants: pd.DataFrame, detalle_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if long.empty:
        return outputs

    cats = list(_data.VALIDATION_CHOICES)

    # --- accuracy estricta por categoría (standalone) ---
    sub = long[long["format"] == "classification"]
    by_cat = (
        sub.groupby("category")
        .agg(n=("is_correct_strict", "size"), correct=("is_correct_strict", "sum"))
    )
    by_cat["acc"] = (by_cat["correct"] / by_cat["n"] * 100).round(1)
    by_cat = by_cat.reindex(cats, fill_value=0)
    labels = [f"{P.BUCKET_LABELS_ES[c]}\n(n={int(by_cat.loc[c, 'n'])})" for c in cats]
    palette = {P.BUCKET_LABELS_ES[c]: P.BUCKET_COLORS[c] for c in cats}
    counts_series = pd.Series(by_cat["acc"].values, index=labels)
    path = P.detail_vertical_bar(
        counts_series, detalle_dir / "accuracy_por_categoria.png",
        color=P.C_VALIDADOS, ylabel="accuracy estricta (%)",
        value_fmt="%.1f", reference=100.0,
        rotate_xticks=20,
    )
    outputs.append(_emit_detail(path, "accuracy_por_categoria",
        f"Accuracy estricta del test de entrada agregada por categoría "
        f"(versión standalone del panel inferior de la matriz de "
        f"confusión). La línea discontinua marca el 100\\%."
    ))

    # --- progresión por número de intento ---
    per_attempt = (
        long.groupby(["username", "attempt"])["points_earned"]
        .sum().reset_index(name="raw_score")
    )
    grouped = per_attempt.groupby("attempt").agg(
        mean_score=("raw_score", "mean"),
        n_users=("username", "nunique"),
    ).reset_index().sort_values("attempt")
    if not grouped.empty:
        ser = pd.Series(
            grouped["mean_score"].round(2).values,
            index=[f"intento {int(a)}\n(n={int(n)})" for a, n in zip(grouped["attempt"], grouped["n_users"])],
        )
        path = P.detail_vertical_bar(
            ser, detalle_dir / "progresion_por_intento.png",
            color=P.C_VOTADOS, ylabel="puntaje medio (sobre 16)",
            value_fmt="%.2f", reference=_data.TEST_PASS_THRESHOLD,
        )
        outputs.append(_emit_detail(path, "progresion_por_intento",
            f"Puntaje medio total por número de intento. Permite "
            f"contestar si los reintentos están actuando como un bucle "
            f"de aprendizaje (puntaje sube) o si los usuarios se "
            f"quedan atascados (puntaje plano). La línea discontinua "
            f"marca el umbral de aprobación "
            f"(\\texttt{{TEST\\_PASS\\_THRESHOLD = {_data.TEST_PASS_THRESHOLD:g}}})."
        ))

    # --- tasa de aprobación por idioma ---
    if participants is not None and not participants.empty:
        rate = (
            participants.groupby("language")
            .apply(lambda g: float((g["test_score"].apply(_data.parse_test_score)
                                      .apply(lambda d: bool(d) and max(d.values()) >= _data.TEST_PASS_THRESHOLD)).mean() * 100), include_groups=False)
            .round(1).sort_index()
        )
        counts_lang = participants.groupby("language").size()
        rate.index = [f"{P.LANGUAGE_LABELS_ES.get(k, k)} (n={int(counts_lang[k])})" for k in rate.index]
        path = P.detail_vertical_bar(
            rate, detalle_dir / "tasa_aprobacion_por_idioma.png",
            color=P.C_APROBADO, ylabel="% participantes",
            value_fmt="%.1f", reference=100.0,
        )
        outputs.append(_emit_detail(path, "tasa_aprobacion_por_idioma",
            f"Porcentaje de participantes que aprobaron el test, "
            f"separado por idioma de la interfaz. Espejo del plot "
            f"correspondiente de la sección 1, pero específico al "
            f"resultado del test."
        ))

    # --- distribución de número de intentos por usuario ---
    if participants is not None and not participants.empty:
        n_attempts = participants["test_score"].apply(_data.parse_test_score).apply(len)
        attempts_counts = n_attempts.value_counts().sort_index()
        attempts_counts.index = [f"{int(k)} intento{'s' if k != 1 else ''}" for k in attempts_counts.index]
        path = P.detail_vertical_bar(
            attempts_counts, detalle_dir / "intentos_por_usuario.png",
            color=P.C_HIGHLIGHT, ylabel="participantes",
        )
        outputs.append(_emit_detail(path, "intentos_por_usuario",
            f"Cuántos intentos del test ha completado cada participante. "
            f"Una cola larga en el lado derecho indica usuarios que "
            f"intentan repetidamente sin progreso (señal de frustración)."
        ))

    # --- accuracy MCQ por pregunta (sólo si hay >=3 MCQs) ---
    sub_mcq = long[long["format"] == "multiple_choice"]
    n_distinct = sub_mcq["question_id"].nunique() if not sub_mcq.empty else 0
    if n_distinct >= MCQ_PLOT_MIN_ITEMS:
        table = (
            sub_mcq.groupby("question_id")
            .agg(n=("is_correct_strict", "size"), correct=("is_correct_strict", "sum"))
            .reset_index()
        )
        table["acc"] = (table["correct"] / table["n"] * 100).round(1)
        table = table.sort_values("acc")
        ser = pd.Series(
            table["acc"].values,
            index=[f"{qid} (n={int(n)})" for qid, n in zip(table["question_id"], table["n"])],
        )
        path = P.detail_horizontal_bar(
            ser, detalle_dir / "accuracy_mcq_por_pregunta.png",
            color=P.C_HIGHLIGHT, xlabel="accuracy estricta (%)",
        )
        outputs.append(_emit_detail(path, "accuracy_mcq_por_pregunta",
            f"Accuracy por pregunta de opción múltiple (formato "
            f"\\texttt{{multiple\\_choice}}). Sólo se emite si el banco "
            f"tiene al menos {MCQ_PLOT_MIN_ITEMS} items MCQ."
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

    if participants_df is None:
        participants_df = _data.load_participants_df()
    long = dl.load_test_responses_long(participants_df)
    summary = build_summary(long)

    csv_path = io_utils.save_csv(summary, out_dir / "test.csv")
    n_responses = len(long)
    n_users = long["username"].nunique() if not long.empty else 0
    n_clas_questions = (
        long[long["format"] == "classification"]["question_id"].nunique()
        if not long.empty else 0
    )
    table_caption = (
        f"Rendimiento agregado del test de entrada por categoría sobre "
        f"$n={n_responses}$ respuestas de $n={n_users}$ participantes "
        f"distintos a $n={n_clas_questions}$ preguntas de clasificación. "
        f"\\textit{{Accuracy estricta}} mide cuándo el participante eligió "
        f"exactamente el bucket correcto, mientras que \\textit{{\\% mismo "
        f"lado}} mide cuándo al menos identificó si la pregunta debía "
        f"ser aceptada o rechazada (incluso si confundió el bucket dentro "
        f"de ese lado). El \\textit{{puntaje medio}} aplica el esquema de "
        f"crédito parcial definido en \\texttt{{test\\_data.grade}}: $+1$ "
        f"para coincidencia exacta, $+0.5$ para mismo lado, $-0.5$ para "
        f"lado contrario."
    )
    latex_utils.save_table_tex(csv_path, summary, table_caption, label="test")
    print(f"wrote {csv_path}  ({len(summary)} filas)")

    confusion_path = plot_matriz_confusion(long, plots_dir)
    n_passed = (
        participants_df["test_score"].apply(_data.parse_test_score)
        .apply(lambda d: bool(d) and max(d.values()) >= _data.TEST_PASS_THRESHOLD).sum()
        if participants_df is not None and not participants_df.empty else 0
    )
    latex_utils.save_figure_tex(
        confusion_path,
        caption=(
            f"Matriz de confusión del test de entrada y accuracy por "
            f"categoría. Panel superior: cada fila corresponde al bucket "
            f"correcto y cada columna al bucket elegido por el "
            f"participante; los valores están normalizados por fila para "
            f"que cada fila sume 100\\%. La diagonal indica respuestas "
            f"exactas; las celdas fuera de la diagonal revelan qué "
            f"buckets se confunden con cuáles — un patrón útil para "
            f"detectar ambigüedades en las guías de anotación. Panel "
            f"inferior: accuracy estricta agregada por categoría. "
            f"$n={n_passed}$ participantes han alcanzado el umbral de "
            f"aprobación de {_data.TEST_PASS_THRESHOLD:g} puntos sobre 16."
        ),
        label="matriz_confusion",
    )
    print(f"wrote {confusion_path}")

    dificultad_path = plot_dificultad_preguntas(long, plots_dir)
    latex_utils.save_figure_tex(
        dificultad_path,
        caption=(
            f"Dificultad por pregunta del test de entrada. Cada barra "
            f"corresponde a una pregunta de clasificación; la longitud "
            f"de la barra indica la accuracy estricta del conjunto de "
            f"participantes que la respondió. El color codifica la "
            f"categoría correcta. Las preguntas aparecen ordenadas de "
            f"menor a mayor accuracy, de modo que los items más "
            f"ambiguos (y por tanto candidatos a reescritura o rotación "
            f"fuera del banco) quedan visualmente al principio de la "
            f"lista — mismo criterio metodológico que el análisis del "
            f"test 2025 documentado en "
            f"\\texttt{{data/analysis\\_test\\_2025.md}}."
        ),
        label="dificultad_preguntas",
    )
    print(f"wrote {dificultad_path}")

    detail_paths = emit_detail_plots(long, participants_df, detalle_dir)
    for p in detail_paths:
        print(f"wrote {p}")

    return [csv_path, confusion_path, dificultad_path, *detail_paths]


def main() -> None:
    run()


if __name__ == "__main__":
    main()
