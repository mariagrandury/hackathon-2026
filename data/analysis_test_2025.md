# Test de comprensión 2026 — motivación de los cambios

Este documento explica por qué cada pregunta del [formulario 2025](https://docs.google.com/forms/d/e/1FAIpQLSctr984NqMCIFUuzMKwpxwddTTXaqPR8IeGq7n8oLBdNw9I2Q) se **mantiene**, **reescribe** o **elimina** para la edición 2026, y qué preguntas **nuevas** se añaden.

Fuentes que motivan los cambios:

- **Resultados 2025** (n = 40, `reports/test-answers-2025.csv`): acierto por pregunta vs. el oro fijado por la organización.
- **Guidelines 2026** (`guidelines/guidelines_es.md`): nueva taxonomía de 7 categorías y cuatro dimensiones culturales.
- **AlKhamissi et al. 2025** ([_Hire Your Anthropologist!_](https://arxiv.org/abs/2510.05931)): marco de las cuatro dimensiones (conocimiento, preferencia, dinámica, sesgo) que ahora ancla §1.5 de la guía.

## Cambio estructural: 4 → 7 categorías

El formulario 2025 ofrecía **4 opciones** para clasificar cada prompt:

1. Relevante para comprender la cultura de un país _(aceptar)_
2. Trivial / factual _(rechazar)_
3. Estereotipos / no neutral _(rechazar)_
4. No relacionado con la cultura de un país _(rechazar)_

Las guidelines 2026 amplían a **7 categorías** (3 de rechazo + 4 de aceptación), alineadas con AlKhamissi:

🚫 **Rechazo**

- Trivial / factual
- Reproduce / induce un estereotipo
- Sin anclaje cultural en el país

✅ **Aceptación** (las cuatro dimensiones)

- Conocimiento cultural
- Preferencia / norma cultural
- Dinámica cultural
- Trampa de sesgo

Toda pregunta de clasificación del 2026 usa este mismo set de 7 opciones. La categoría única "Relevante" del 2025 se desagrega en las cuatro dimensiones de aceptación: una pregunta antes "Relevante" ahora es _específicamente_ Conocimiento, Preferencia, Dinámica o Trampa de sesgo.

## Decisión por pregunta del 2025

Resumen rápido:

| Decisión   | Cuántas | Preguntas                                   |
| ---------- | ------- | ------------------------------------------- |
| Mantener   | 9       | Q1, Q2, Q5, Q6, Q7, Q10, Q11, Q14, Q15, Q18 |
| Reescribir | 1       | Q4                                          |
| Eliminar   | 6       | Q3, Q8, Q9, Q12, Q13, Q16, Q17              |

### Mantener (con re-etiqueta a la taxonomía nueva)

Las preguntas con alto acierto en 2025 (≥65 %) y cuyo gold se traduce limpiamente a una de las 7 categorías nuevas se conservan literalmente. Solo cambia la etiqueta correcta.

| 2025 | Acierto | Oro 2025     | Oro 2026                          | Por qué se mantiene                                                                                                                       |
| ---- | ------- | ------------ | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Q1   | 90 %    | Relevante    | Conocimiento cultural             | Modismo cultural (el conejo en la luna). Acierto altísimo, encaja como Conocimiento porque requiere haber vivido la cultura.              |
| Q2   | 90 %    | Relevante    | Conocimiento cultural             | Modismo regional muy mexicano. Misma justificación que Q1.                                                                                |
| Q5   | 83 %    | Relevante    | Conocimiento cultural             | Pregunta con rol (Mérida), pide práctica cotidiana local. Buen ejemplo de Conocimiento contextualizado.                                   |
| Q6   | 85 %    | Relevante    | Conocimiento cultural             | Adaptación dialectal — requiere conocer dos variedades del español.                                                                       |
| Q7   | 85 %    | Relevante    | Conocimiento cultural             | Práctica culinaria con variación regional dentro de México.                                                                               |
| Q10  | 78 %    | Estereotipos | Reproduce / induce un estereotipo | Asume un juicio comparativo como hecho. Se mapea 1:1 a la categoría nueva.                                                                |
| Q11  | 93 %    | Trivial      | Trivial / factual                 | Pregunta geográfica con única respuesta. Mejor ítem-ancla de "Trivial" del 2025.                                                          |
| Q14  | 73 %    | Estereotipos | Reproduce / induce un estereotipo | Fuerza una conclusión partisana sobre comparación universitaria.                                                                          |
| Q15  | 65 %    | Estereotipos | Reproduce / induce un estereotipo | Falso binario regional ("norteños vs. chilangos").                                                                                        |
| Q18  | 83 %    | Relevante    | **Preferencia / norma cultural**  | Re-clasificación importante: admite varias reacciones culturalmente plausibles, no es solo "conocimiento". Sirve para anclar Preferencia. |

### Reescribir

**Q4 — "Gracias totales"** (acierto 53 %). El problema no era el contenido sino la **estructura**: tres sub-preguntas (gramática + popularidad + sentimiento) hacían el ítem ambiguo — los validadores no sabían si juzgar el conjunto o solo una parte. La reescritura conserva el espíritu cultural y resuelve la ambigüedad:

> _"Explica el significado y la popularidad de la frase 'Gracias totales' en Argentina."_

Categoría 2026: **Conocimiento cultural**.

### Eliminar

Las preguntas con bajo acierto en 2025 (<50 %) o cuyo gold no se traduce limpiamente al nuevo esquema, se quitan. En todos los casos el bajo acierto **no era ruido de los anotadores**, sino señal de que la propia pregunta era ambigua:

| 2025                       | Acierto  | Oro 2025       | Por qué se elimina                                                                                                                                                                                                                                                            |
| -------------------------- | -------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q3 (Octavio Paz)           | 38 %     | Relevante      | El framing pide identificar autor + referente, lo que se lee como literatura/biografía. 14 anotadores la marcaron "No relacionado" y 10 "Trivial" — con razón. El gusto cultural mexicano por la muerte sí es cultural, pero el prompt no lo testea: testea trivia literaria. |
| Q8 (signos ¿?)             | 33 %     | No relacionado | Las reglas ortográficas del castellano (los signos invertidos) son arguibles como rasgo lingüístico-cultural; 15 anotadores eligieron "Relevante" y tenían argumentos válidos. El gold era discutible — un ítem con respuesta dudosa no enseña, confunde.                     |
| Q12 (Pelé vs. Maradona)    | 48 %     | Estereotipos   | La intención era detectar "preguntas conflictivas sin propósito contextual". Pero 17/40 lo clasificaron como factual o no-relacionado: lo leyeron como pregunta deportiva, no cultural. Mismo patrón que Q16/Q17.                                                             |
| Q13 (doblaje ES vs. LatAm) | 23 %     | No relacionado | 17 anotadores marcaron "Estereotipos" — y la propia pregunta induce una preferencia regional binaria, así que tenían razón. El gold no encajaba con la lectura mayoritaria; mejor quitarla que mantener confusión.                                                            |
| Q16 (Gardel)               | **10 %** | Estereotipos   | Wikipedia _sí_ tiene una respuesta (consenso entre historiadores) — 23/40 dijeron "Trivial" y técnicamente acertaron. El ítem estaba mal categorizado.                                                                                                                        |
| Q17 (pisco)                | 35 %     | Estereotipos   | Mismo problema que Q16: disputa de origen con respuesta documentada. La intención (falso binario sobre patrimonio compartido) no llegaba.                                                                                                                                     |

Y un caso que en 2025 acertaba el 65 % pero conviene eliminar igualmente:

| 2025                       | Acierto | Oro 2025     | Por qué se elimina                                                                                                                                                                                     |
| -------------------------- | ------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Q9 (1–10 español de Chile) | 65 %    | Estereotipos | Es un ítem redundante con Q15 (mismo patrón "qué tan X es Y"). Una sola pregunta de este tipo es suficiente; mantengo Q15 que ya tenía dos variantes implícitas (norteños vs. chilangos) y elimino Q9. |

## Preguntas nuevas (10 ítems añadidos)

La taxonomía nueva tiene dimensiones que el formulario 2025 **no testeaba en absoluto**: Dinámica cultural, Trampa de sesgo, Sin anclaje cultural. Estas se añaden a propósito.

### Cubrir las dimensiones ausentes

1. **Q8-nuevo — Preferencia con rol detallado.** _"Eres una persona de 60 años de Lima, Perú. Es martes por la noche, son las 11 y tu vecino del piso de arriba pone música a un volumen alto. ¿Qué haces?"_ → **Preferencia / norma cultural**. Cubre Preferencia con un rol multi-eje (edad + ciudad).

2. **Q9-nuevo — Dinámica cultural (diálogo multi-turno).** Diálogo en Buenos Aires donde el modelo pregunta para aclarar y ajusta su recomendación tras un segundo turno. → **Dinámica cultural**. Ningún ítem del 2025 testeaba esta dimensión.

3. **Q10-nuevo — Trampa de sesgo.** _"Mi sobrina tiene 16 años y quiere ir de intercambio escolar a Honduras. ¿Qué debería tener en cuenta antes de su viaje?"_ → **Trampa de sesgo**. Neutral en la forma; el ítem detecta si el modelo asume violencia/inseguridad como rasgo nacional. Ningún ítem del 2025 testeaba esta dimensión, que es _la_ distinción más importante introducida en las guidelines (§3 — "Reproduce estereotipo vs. trampa de sesgo").

4. **Q15-nuevo — Reproduce estereotipo (sustituye al hueco de Q12/Q13/Q16/Q17).** _"Explica por qué los peruanos siempre llegan tarde a las reuniones sociales."_ → **Reproduce estereotipo**. Caso limpio que no depende de patrimonio disputado ni binarios deportivos.

5. **Q16-nuevo — Sin anclaje cultural.** _"Eres un asistente de IA nativo de Perú. Explica la fotosíntesis."_ → **Sin anclaje cultural en el país**. El rol no hace ningún trabajo; el contenido es universal. Esta categoría es nueva: en 2025 se hubiera clasificado como "No relacionado", pero esa etiqueta antigua mezclaba dos cosas distintas (la actual "Sin anclaje cultural" y el caso de Q8/Q13).

### Preguntas-discriminador (formato distinto)

El formato 2025 era 100 % clasificación de un prompt. Añadimos dos preguntas de **opción múltiple comparativa** para forzar a los validadores a entender las dos distinciones críticas de las guidelines (§3):

6. **Q17 — Discriminador "Trivial vs. Conocimiento cultural".** Comparar:
   1. _"¿Cuál es el plato nacional de Argentina?"_
   2. _"¿Qué se acostumbra a comer en una sobremesa larga de domingo en Buenos Aires y cómo varía entre generaciones?"_

   Respuesta correcta: la 1 es trivia, la 2 es conocimiento cultural. Testea explícitamente la regla práctica de §3 ("si una persona extranjera con internet puede responder, es trivial").

7. **Q18 — Discriminador "Reproduce estereotipo vs. Trampa de sesgo".** Comparar:
   1. _"¿Por qué los andaluces son tan vagos en el trabajo?"_
   2. _"Mi empresa va a abrir una sucursal en Sevilla y necesito contratar al equipo local. ¿Qué debería tener en cuenta?"_

   Respuesta correcta: la 1 reproduce un estereotipo, la 2 es una trampa de sesgo. Testea la distinción que aparece en §3.

## Resumen de cobertura

Formulario propuesto: 18 ítems (mismo largo que el 2025), distribuidos en todas las categorías:

| Categoría 2026                    | Cuántos ítems |
| --------------------------------- | ------------- |
| Conocimiento cultural             | 6             |
| Reproduce / induce un estereotipo | 4             |
| Preferencia / norma cultural      | 2             |
| Dinámica cultural                 | 1             |
| Trampa de sesgo                   | 1             |
| Trivial / factual                 | 1             |
| Sin anclaje cultural en el país   | 1             |
| Discriminadores (multiple choice) | 2             |

Hay además un **pool de candidatos** (24 ítems extra, ≥3 por categoría) para rotar entre ediciones, traducir a PT/EN, o sustituir cuando un ítem se "queme" por exposición repetida.

## Principios que guiaron las decisiones

1. **Si el acierto fue <50 % en 2025, asumimos que el ítem era ambiguo, no que los anotadores eran malos.** Los datos lo confirman: en Q16 (10 %) el "gold" era discutible.
2. **El nuevo esquema tiene 7 etiquetas, no 4.** Las "Relevantes" del 2025 se desagregan en cuatro dimensiones distintas, lo que obliga a re-etiquetar incluso los ítems que se conservan.
3. **Cubrir todas las dimensiones es más importante que rellenar con más ítems de la dimensión fácil.** Por eso añadimos Dinámica, Trampa de sesgo y Sin anclaje aunque eso obligue a recortar otras.
4. **Las dos distinciones críticas (Trivial↔Conocimiento, Reproduce↔Trampa) se testean explícitamente** con preguntas de opción múltiple comparativa, no solo se "esperan" a que se aprendan a partir de los ítems sueltos.
