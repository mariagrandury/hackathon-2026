# Hackathon 2026 — suite de análisis

Análisis offline modular de los dos datasets privados de HF más el
export de Eventbrite. Cada sección produce:

- **1 tabla** (CSV resumen + `.tex` con caption y entorno `\begin{table}`).
- **2 figuras HEADLINE** densas y multi-panel (PNG + `.tex` con caption
  y entorno `\begin{figure}`). Son las figuras paper-ready.
- **Plots de detalle** (en `plots/detalle/`, también PNG + `.tex`): una
  figura por dimensión individual del dataset. Cubren toda la
  información de las columnas relevantes para que el informe pueda
  citar cualquier corte. Estilísticamente más simples (un solo chart
  cada una) y útiles también para slides.

Todos los plots y captions están en **español**. Los PNG no llevan
título — la información descriptiva vive en el `.tex` companion.

El **porqué** de cada figura vive en
[ANALYSIS_PROPOSAL.md](ANALYSIS_PROPOSAL.md).

## Layout

```
analysis/
├── ANALYSIS_PROPOSAL.md
├── README.md
├── run_all.py
├── _common/
│   ├── data_loading.py            # cargadores + join con Eventbrite
│   ├── plotting.py                # primitivas de seaborn + paleta
│   ├── metrics.py                 # Wilson CI, Gini, Lorenz, unanimidad
│   ├── latex_utils.py             # generación de .tex (figure, table)
│   └── io_utils.py
├── 01_participant_demographics/
│   ├── analyze.py
│   ├── participantes.csv          # 1 fila por país (resumen)
│   ├── participantes.tex          # tabla LaTeX con caption
│   └── plots/
│       ├── panorama_demografico.png + .tex            # HEADLINE
│       ├── mapa_pais_perfil.png     + .tex            # HEADLINE
│       └── detalle/                                    # cobertura completa
│           ├── participantes_por_pais.png + .tex
│           ├── participantes_por_idioma.png + .tex
│           ├── participantes_pais_x_idioma.png + .tex
│           ├── puntaje_test_distribucion.png + .tex
│           ├── tasa_aprobacion_por_idioma.png + .tex
│           ├── pronombres_distribucion.png + .tex
│           ├── educacion_distribucion.png + .tex
│           ├── campo_estudio_distribucion.png + .tex
│           ├── nivel_nlp_distribucion.png + .tex
│           ├── primera_vez_somosnlp_distribucion.png + .tex
│           └── anio_nacimiento_distribucion.png + .tex
├── 02_prompt_writing/
│   ├── analyze.py
│   ├── prompts.csv                + .tex
│   └── plots/
│       ├── panorama_prompts.png        + .tex          # HEADLINE
│       ├── mapa_pais_idioma_estado.png + .tex          # HEADLINE
│       └── detalle/
│           ├── embudo_pipeline.png + .tex
│           ├── prompts_por_pais.png + .tex
│           ├── prompts_por_idioma.png + .tex
│           ├── prompts_pais_x_idioma.png + .tex
│           ├── longitud_prompt_caracteres.png + .tex
│           ├── longitud_prompt_palabras.png + .tex
│           ├── uso_system_prompt_por_idioma.png + .tex
│           ├── top_autores.png + .tex
│           ├── histograma_prompts_por_autor.png + .tex
│           ├── curva_lorenz.png + .tex
│           └── estado_validacion_por_prompt.png + .tex
├── 03_prompt_validation/
│   ├── analyze.py
│   ├── validacion.csv             + .tex
│   └── plots/
│       ├── panorama_validacion.png + .tex              # HEADLINE
│       ├── mapa_pais_bucket.png   + .tex               # HEADLINE
│       └── detalle/
│           ├── embudo_slots_validacion.png + .tex
│           ├── distribucion_buckets.png + .tex
│           ├── buckets_por_idioma.png + .tex
│           ├── buckets_por_pais.png + .tex
│           ├── tasa_aceptacion_por_pais.png + .tex
│           ├── top_validadores.png + .tex
│           ├── tasa_unanimidad_por_idioma.png + .tex
│           └── acuerdo_aceptacion_rechazo.png + .tex
├── 04_answer_voting/
│   ├── analyze.py
│   ├── votacion.csv               + .tex
│   └── plots/
│       ├── panorama_votacion.png      + .tex           # HEADLINE
│       ├── mapa_modelo_categoria.png  + .tex           # HEADLINE
│       └── detalle/
│           ├── embudo_votacion.png + .tex
│           ├── distribucion_elecciones.png + .tex
│           ├── tasa_victoria_modelos.png + .tex
│           ├── tasa_indecisos_modelos.png + .tex
│           ├── votos_por_categoria.png + .tex
│           ├── votos_por_pais.png + .tex
│           └── top_votantes.png + .tex
└── 05_entry_test/
    ├── analyze.py
    ├── test.csv                   + .tex
    └── plots/
        ├── matriz_confusion.png      + .tex            # HEADLINE
        ├── dificultad_preguntas.png  + .tex            # HEADLINE
        └── detalle/
            ├── accuracy_por_categoria.png + .tex
            ├── progresion_por_intento.png + .tex
            ├── tasa_aprobacion_por_idioma.png + .tex
            ├── intentos_por_usuario.png + .tex
            └── accuracy_mcq_por_pregunta.png + .tex  (sólo si MCQs >= 3)
```

Algunos plots de detalle se emiten condicionalmente — si no hay
suficientes datos para que sean informativos (e.g., sin votos
registrados, o MCQs < 3), el script los omite sin error.

## Requisitos

Mismo entorno que el Space — ver `CLAUDE.md` en la raíz. En resumen:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib seaborn   # no están pinned en requirements.txt
cp .env.example .env             # con un token HF de lectura
```

El análisis es de **sólo lectura** contra el Hub.

## Correr todas las secciones

```bash
python analysis/run_all.py
```

Carga los dos datasets de HF una sola vez y los pasa a cada sección, así
el Hub se golpea dos veces total (no 5).

## Correr una sección aislada

Los nombres de carpeta empiezan con dígito, lo que impide la sintaxis
`python -m analysis.01_...`. Se usa la forma script:

```bash
python analysis/01_participant_demographics/analyze.py
python analysis/02_prompt_writing/analyze.py
python analysis/03_prompt_validation/analyze.py
python analysis/04_answer_voting/analyze.py
python analysis/05_entry_test/analyze.py
```

## Cómo usar los `.tex` en un informe

Cada figura y cada tabla viene con su propio `.tex` listo para incluir:

```latex
\input{analysis/01_participant_demographics/plots/panorama_demografico.tex}
\input{analysis/01_participant_demographics/participantes.tex}
```

El `.tex` ya contiene `\begin{figure}` / `\begin{table}`, el `\caption{}`
y el `\label{}`. El paquete `booktabs` y `graphicx` son los únicos
requisitos en el preámbulo del informe.

Las **figuras no tienen título** dentro del PNG — toda la información
descriptiva está en el caption del `.tex`. Esto evita duplicación cuando
la figura va dentro de un informe y permite reutilizar el PNG en slides
sin redibujar.

## Variables de entorno

- `HACKATHON_EVENTBRITE_CSV=/ruta/al/report.csv` — anula el
  auto-descubrimiento del `reports/report-*.csv` más reciente.
- `HACKATHON_LOG_LEVEL=WARNING` — silencia logs de cache-miss /
  conflicto de commit mientras corre el suite.

## Reproducibilidad

Los CSVs se comprometen junto con los PNGs y `.tex`, así un lector sin
acceso al Hub puede reconstruir las tablas y los plots desde la misma
fuente. Volver a correr `analyze.py` sobrescribe los artefactos
existentes.
