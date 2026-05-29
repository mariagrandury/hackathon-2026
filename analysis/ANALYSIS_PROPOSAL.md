# Propuesta de análisis — Hackathon 2026

Este documento explica **qué** muestra cada plot del suite de análisis y,
sobre todo, **por qué** se eligió. Está en español porque el público
objetivo del informe final lo está; los nombres de archivo del código
siguen en inglés (identificadores estables).

Cada sección produce:
- **1 tabla resumen** (CSV + `.tex` companion con caption y entorno
  `\begin{table}` listo para `\input`).
- **2 figuras HEADLINE** densas y multi-panel (PNG + `.tex` companion).
  Son las paper-ready, pensadas para encajar en el informe principal.
- **Plots de detalle** en `plots/detalle/` que cubren cada dimensión
  individual del dataset (un chart por figura, formato sencillo). Cada
  plot de detalle también lleva su `.tex` companion con caption. Útiles
  como anexos del informe o como slides individuales.

Los plots no llevan título — la información descriptiva vive en la
**caption del `.tex`** para evitar duplicación con el cuerpo del informe.

## Metodología

- **Fuentes de datos.** Tres: los dos datasets privados de HF
  (`hackathon_participants`, `cultural_preferences`) cargados vía
  `data.load_*_df()`; el registro más reciente de Eventbrite en
  `reports/report-*.csv` para enriquecer los datos demográficos
  (pronombres, educación, campo de estudio, nivel de PLN, primera vez,
  año de nacimiento). La unión Eventbrite ↔ HF se hace por **usuario
  de HF en minúsculas**.
- **Exclusiones.** Las filas con autor `"v0"` (~2k prompts importados
  del dataset 2025) se mantienen en los datos pero **se marcan** con
  `is_v0_import`. Quedan excluidas de los plots centrados en autores
  (top, Lorenz) para no inflar las estadísticas, pero **se cuentan**
  en métricas de pipeline (volumen por país, por idioma, embudo del
  ciclo de vida) porque son prompts reales que sí avanzan al voto.
- **Forma de los CSV resumen.** Una fila por agrupación natural (país,
  bucket, modelo, categoría), no por entidad individual. El CSV de la
  sección \emph{es} la tabla; el `.tex` companion la formatea con
  `booktabs` para incluir directamente en LaTeX.
- **Sin línea temporal.** Los datasets de HF no almacenan timestamps
  por prompt / validación / voto, así que el suite **no tiene** plots
  de evolución temporal.
- **Cuidados estadísticos.** El hackathon terminará con miles bajos de
  prompts y cientos bajos de votantes por modelo. El plot de tasas de
  victoria usa **intervalo de confianza de Wilson al 95%** para evitar
  conclusiones sobre muestras pequeñas. Los mapas de calor enmascaran
  países con <3 participantes o <5 prompts para que las celdas con N
  bajo no introduzcan ruido visual.

## Sección 1 — Demografía de participantes

**Pregunta clave.** ¿Quién está participando y está balanceado el
cohorte?

- **Figura 1: `panorama_demografico.png`** (composición 2×2)
  - **Países × aprobación del test** (countplot con hue): combina dos
    señales en una. Permite ver a la vez la cobertura geográfica y qué
    países están aprobando más el test.
  - **KDE del puntaje del test por idioma**: tres densidades superpuestas
    revelan si el test está calibrado igual entre idiomas (línea
    vertical en el umbral = 12). Si EN o PT tienen densidad muy a la
    izquierda, la traducción del banco tiene bugs.
  - **Donut de pronombres**: composición del cohorte. Donut > pie
    porque el legend lateral evita amontonar etiquetas.
  - **Barras polares de nivel de PLN**: visualización circular que
    enfatiza la composición proporcional (básico / intermedio / avanzado).
- **Figura 2: `mapa_pais_perfil.png`** (heatmap)
  - País × dimensión demográfica (% participantes del país en cada
    categoría). Filas ordenadas por % aprobó test descendente, así
    saltan a la vista patrones del tipo "los países con mejor
    rendimiento tienden a tener más PhDs".

## Sección 2 — Escritura de prompts

**Pregunta clave.** ¿Quién escribe, cuánto y dónde se atasca el pipeline?

- **Figura 1: `panorama_prompts.png`** (composición 2×2)
  - **Embudo del ciclo de vida**: escritos → totalmente validados → con
    respuestas → totalmente votados. La gráfica titular del informe.
  - **Curva de Lorenz + Gini**: equidad de la contribución. Gini ≈ 0
    sería balanceado; Gini → 1 sería un único autor escribiendo todo.
    Métrica crítica para citar el dataset.
  - **Violinplot de longitud por idioma**: prompts muy cortos son
    perezosos; muy largos son off-task. La forma del violín revela
    diferencias estilísticas entre idiomas.
  - **Top 15 autores**: identifica super-contribuidores y posible
    burn-out.
- **Figura 2: `mapa_pais_idioma_estado.png`** (dos heatmaps lado a lado)
  - Izquierda: país × idioma (conteo absoluto). Detecta desajustes
    "país X escribiendo en EN" que apuntan a strings de UI faltantes.
  - Derecha: país × estado de validación (0/1/2/3 slots, normalizado
    por fila). Muestra dónde el pipeline tiene volumen pero falta
    validación, y dónde le falta volumen sin más.

## Sección 3 — Validación de prompts

**Pregunta clave.** ¿Qué se acepta, dónde y los validadores coinciden?

- **Figura 1: `panorama_validacion.png`** (composición 2×2)
  - **Barras polares por bucket**: distribución absoluta de validaciones
    entre los 7 buckets de la taxonomía. Los buckets de rechazo en tonos
    cálidos, los de aceptación en tonos fríos. Visualización circular
    permite leer la composición de un vistazo.
  - **Embudo de slots por prompt**: cuántos prompts tienen 0/1/2/3 slots
    de validación llenos. Indica dónde está atascado el pipeline.
  - **Donut aceptación / rechazo**: reparto agregado del lado.
  - **Tasa de unanimidad por idioma**: % de prompts totalmente validados
    en los que los tres validadores coincidieron en el bucket. La
    métrica de fiabilidad más honesta dado N (no usamos Cohen's kappa /
    Krippendorff's alpha — la pareja de validadores no es sistemática y
    la muestra es pequeña).
- **Figura 2: `mapa_pais_bucket.png`** (heatmap)
  - País × bucket de validación (conteo absoluto). Revela qué dimensiones
    culturales concentran la atención en cada región — sirve para
    detectar tanto sesgos de cobertura como sesgos de dimensión.

## Sección 4 — Votación de respuestas

**Pregunta clave.** ¿Qué modelos manejan mejor prompts culturalmente
arraigados? ¿Qué nos dicen los votos "ambas"/"ninguna"?

- **Figura 1: `panorama_votacion.png`** (composición 2×2)
  - **Embudo de votación**: con respuestas → 1+ votos → 2+ → 3 votos.
  - **Donut de elección** (a / b / ambas / ninguna): la **tasa de
    indecisos** ("ambas" + "ninguna") es la señal de calidad menos
    explotada del pipeline A/B — alta = prompts no discriminantes.
  - **Forest plot con IC95 de Wilson**: tasa de victoria por modelo. El
    output de investigación principal. IC95 previene afirmar de más
    cuando N es bajo.
  - **% indecisos por modelo**: qué modelos quedan emparejados en
    comparaciones poco interesantes (hallucinan idéntico al oponente
    o son indistinguibles).
- **Figura 2: `mapa_modelo_categoria.png`** (heatmap)
  - Modelo elegido × bucket de validación dominante del prompt. Revela
    **especialización**: un modelo que domina en *conocimiento* pero
    cae en *trampa de sesgo* es un hallazgo sustantivo, no un promedio.

## Sección 5 — Test de entrada

**Pregunta clave.** ¿Están calibradas las preguntas? ¿Funciona la
taxonomía de 7 buckets?

- **Figura 1: `matriz_confusion.png`** (composición vertical 4:1.5)
  - **Arriba: matriz de confusión** bucket correcto × bucket elegido
    (normalizada por fila). La diagonal indica respuestas exactas;
    las celdas fuera de la diagonal revelan qué buckets se confunden
    con cuáles — patrón para detectar ambigüedades en las guías.
  - **Abajo: accuracy por categoría**. Cierra la matriz con la métrica
    agregada por bucket.
- **Figura 2: `dificultad_preguntas.png`** (barras horizontales coloreadas)
  - Una barra por pregunta de clasificación, ordenadas por accuracy
    ascendente. Color = categoría correcta. Los items más ambiguos
    quedan visualmente al principio de la lista — candidatos a rotar.
    Mismo criterio metodológico que `data/analysis_test_2025.md`.

## Lo que el suite intencionalmente NO cubre

- **Ninguna métrica temporal.** No hay timestamps.
- **Acuerdo inter-anotador más allá de la unanimidad.** Cohen's kappa
  / Krippendorff's alpha necesitan diseño cruzado, que no tenemos —
  unanimidad y acuerdo de lado son honestos con el N disponible.
- **Heatmap head-to-head de modelos.** El emparejamiento es aleatorio;
  celdas dispersas. El forest plot es la forma correcta para estos
  datos.
- **Topic modeling del contenido.** Fuera de alcance.

## Próximos pasos

1. **Cruz demografía × calidad** (sección 06 futura). ¿Nivel de PLN
   correlaciona con tasa de aceptación? ¿Campo de estudio correlaciona
   con distribución de buckets?
2. **Dashboard de cuotas por (idioma, país)** para organizadores.
3. **Preguntas de quality-check ocultas** (ver `test_data.load_hidden_questions`).
   Cuando aterricen, sus respuestas entran en el mismo esquema de
   `test_responses` y la sección 05 las cubre sin cambios.
