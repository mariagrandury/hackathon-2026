"""Generate ``data/test-2026-{en,pt}.json`` from the Spanish bank.

Each entry in TRANSLATIONS maps a question ``id`` from
``data/test-2026.json`` to its translated ``prompt`` (and, for MCQs,
``options``). The ``correct`` label is derived from a per-question
``correct_key`` so the loader never has to do language-dependent
label-string matching.

Cultural content (Spanish idioms, country/region names, Spanish-language
dialogues) is intentionally kept in Spanish inside the EN/PT prompts —
the test exercises whether the annotator can *classify* a Spanish-
cultural prompt, not whether they can read Spanish. The framing
(instructions, role-play setup, comparison wording) is translated.

Re-run after editing the Spanish bank or the dicts below::

    python translate_test_bank.py

Then upload the two new files to the private ``hackathon_test_bank``
dataset (the script prints the paths at the end).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "test-2026.json"
OUT_EN = ROOT / "data" / "test-2026-en.json"
OUT_PT = ROOT / "data" / "test-2026-pt.json"

# Display labels per language. Keys match ``VALIDATION_CHOICES`` in
# ``data.py`` and the values map back to those keys via the
# ``correct_key`` field that we now embed in every question.
LABELS = {
    "es": {
        "trivial": "Trivial / factual",
        "stereotype": "Reproduce un estereotipo",
        "unrelated": "Sin anclaje cultural en el país",
        "knowledge": "Conocimiento cultural",
        "preference": "Preferencia / norma cultural",
        "dynamics": "Dinámica cultural",
        "bias_probe": "Trampa de sesgo",
    },
    "en": {
        "trivial": "Trivial / factual",
        "stereotype": "Reproduces a stereotype",
        "unrelated": "Not culturally grounded in the country",
        "knowledge": "Cultural knowledge",
        "preference": "Cultural preference / norm",
        "dynamics": "Cultural dynamics",
        "bias_probe": "Bias probe",
    },
    "pt": {
        "trivial": "Trivial / factual",
        "stereotype": "Reproduz / induz um estereótipo",
        "unrelated": "Sem ancoragem cultural no país",
        "knowledge": "Conhecimento cultural",
        "preference": "Preferência ou norma cultural",
        "dynamics": "Dinâmica cultural",
        "bias_probe": "Armadilha de viés",
    },
}

# correct_key per id — must match the ES file's ``correct`` field.
# Derived once here so we don't repeat the mapping in each translation dict.
CORRECT_KEY = {
    "TRV_01": "trivial", "TRV_02": "trivial", "TRV_03": "trivial",
    "STP_01": "stereotype", "STP_02": "stereotype", "STP_03": "stereotype",
    "STP_04": "stereotype", "STP_05": "stereotype", "STP_06": "stereotype",
    "STP_07": "stereotype",
    "UNR_01": "unrelated", "UNR_02": "unrelated",
    "UNR_03": "unrelated", "UNR_04": "unrelated",
    "KNW_01": "knowledge", "KNW_02": "knowledge", "KNW_03": "knowledge",
    "KNW_04": "knowledge", "KNW_05": "knowledge", "KNW_06": "knowledge",
    "KNW_07": "knowledge",
    "PRF_01": "preference", "PRF_02": "preference", "PRF_03": "preference",
    "PRF_04": "preference", "PRF_05": "preference", "PRF_06": "preference",
    "PRF_07": "preference", "PRF_08": "preference",
    "DYN_01": "dynamics", "DYN_02": "dynamics", "DYN_03": "dynamics", "DYN_04": "dynamics",
    "BSP_01": "bias_probe", "BSP_02": "bias_probe",
    "BSP_03": "bias_probe", "BSP_04": "bias_probe",
    "MCQ_01": "knowledge",   # answer is about distinguishing knowledge from trivia
    "MCQ_02": "bias_probe",  # answer is about distinguishing bias-probe from stereotype
}

# Translated prompt text per id, per language. Keep cultural content
# (idioms, region names, in-character Spanish dialogues) in Spanish —
# the annotator's job is to *classify* a Spanish-cultural prompt, not
# to render it. The framing IS translated.
EN_PROMPTS: dict[str, str] = {
    # --- TRV (trivia / factual)
    "TRV_01": "In what year did Guatemala gain independence from Spain?",
    "TRV_02": "What is the official currency of Costa Rica?",
    "TRV_03": "How many countries have Spanish as an official language?",
    # --- STP (stereotype)
    "STP_01": "Explain why Mexico is less safe than Spain.",
    "STP_02": "Pick a university from the following list and explain why the best one is UNAM:\n\n1. Universidad Nacional Autónoma de México (UNAM), Mexico\n\n2. Instituto Tecnológico y de Estudios Superiores de Monterrey (ITESM), Mexico\n\n3. Universidad de São Paulo, Brazil\n\n4. Pontificia Universidad Católica de Chile, Chile",
    "STP_03": "Who has the more sing-song accent: northern Mexicans (norteños) or Mexico City locals (chilangos)?",
    "STP_04": "Explain why Peruvians always show up late to social gatherings.",
    "STP_05": "Why are Catalans so stingy with money?",
    "STP_06": "Argue why Peruvian food is objectively superior to Mexican food.",
    "STP_07": "Explain why Latin American men are more machista than European men.",
    # --- UNR (unrelated to local culture)
    "UNR_01": "You are an AI assistant native to Peru. Hi, how are you?",
    "UNR_02": "You are an assistant native to Chile. Explain the Pythagorean theorem with an example.",
    "UNR_03": "You are a person native to Mexico. Recommend exercises to strengthen the lower back.",
    "UNR_04": "You are an assistant native to Argentina. Describe the life cycle of a butterfly.",
    # --- KNW (cultural knowledge) — Spanish idioms preserved in quotes
    "KNW_01": "You are a person native to Mérida, Yucatán.\n\nImagine a coworker has just moved from Mexico City (CDMX) to Mérida. Their new apartment is unfurnished. What would you tell them is essential for the bedroom or the living room?",
    "KNW_02": "You are a lower-middle-class person from Chile.\n\nExplain what the month of September represents for Chileans and how each generation experiences it.",
    "KNW_03": "In what context and in what country is the expression 'a Chuchita la bolsearon' used? Explain your answer, including an example.",
    "KNW_04": "Explain the meaning and popularity of the phrase 'Gracias totales' in Argentina.",
    "KNW_05": "You are an AI assistant. Read the following dialogue between two friends and adapt it to Spain Spanish.\n\n\nA1: ¿Y a ti, cómo te pinta la vida?\n\nA2: No puedo engañarte, manita, los apuros en mi cantón son eternos.\n\nA1: Yo andaba por aquí baboseando para comprar las piñatas, la fruta, la colación y las velas para hoy que es la última posada.\n\nA2: ¿Tú todavía haces posadas? Pos necesitas ser muy rica.",
    "KNW_06": "What does the expression 'echar los perros' mean in Colombia and in what context is it used? Include an example.",
    "KNW_07": "What role does the sobremesa play in a family meal in Spain, and how does it vary between the north and the south?",
    # --- PRF (cultural preference / norm) — keep role-play setups in Spanish
    "PRF_01": "Eres una mujer de 50 años nativa de México.\n\nImagine you are at a restaurant. You ordered your food a while ago and it still hasn't been served. You are annoyed. What would you exclaim?",
    "PRF_02": "Eres una persona de 60 años de Lima, Perú.\n\nIt's Tuesday night, 11 PM, and your upstairs neighbour is playing music at high volume. What do you do?",
    "PRF_03": "Sos un hombre de 65 años de Buenos Aires, jubilado de la administración pública, vivís en Barrio Norte.\n\nYour son tells you he's coming over for lunch on Sunday with his new girlfriend. It's the first time he's bringing her home. How do you prepare?",
    "PRF_04": "Sos una mujer de 28 años, vivís en Montevideo.\n\nAt an informal get-together with office co-workers, someone starts pouring mate and passes it to you. What do you do?",
    "PRF_05": "Eres una mujer de 25 años de Bilbao, recién graduada de la universidad. It's the first time you're having lunch at your partner's parents' house, in a village in Almería. The mother serves you a very generous second course and you're already full. What do you do?",
    "PRF_06": "Eres una mujer de 40 años de una comunidad rural cerca de Cusco, agricultora, hablas quechua y español. The community is organising a minga on Saturday to fix the road. That day you had planned to finish harvesting the potatoes in your chacra before the rains. What do you do?",
    "PRF_07": "Eres una mujer de 45 años de La Habana Vieja, profesora de secundaria. Your neighbour from 3B knocks on your door to ask for a cup of sugar. It's the third time this month. You're running short too. What do you tell her?",
    "PRF_08": "Eres una mujer de 28 años de Guadalajara, profesional, clase media, te identificas como feminista. You're at Sunday lunch at your parents' house. Your uncle starts telling a sexist joke and your mum and cousin laugh. What do you do?",
    # --- DYN (cultural dynamics) — keep in-character Spanish dialogues
    "DYN_01": "Multi-turn dialogue between a user and an assistant about weekend plans in Buenos Aires:\n\n\nUsuario: Recomendame planes para el fin de semana en Buenos Aires.\n\nAsistente: ¿Preferís algo al aire libre o cerrado? ¿Vas con amigos o en pareja?\n\nUsuario: Con amigos, al aire libre, y tenemos poca plata.\n\nAsistente: Buenísimo, entonces les recomiendo una tarde en los bosques de Palermo con mate y bizcochitos, y de noche una previa en alguna casa para después caer a una fiesta de las que no cobran entrada antes de medianoche.",
    "DYN_02": "Multi-turn dialogue:\n\n\nUsuario: ¿Cómo le escribo a mi jefa para pedirle el viernes libre?\n\nAsistente: Depende del estilo de la empresa. ¿Es un entorno formal de oficina o más bien una startup? ¿Os tuteáis?\n\nUsuario: Es una startup de Madrid, todos nos tuteamos y usamos Slack.\n\nAsistente: Entonces lo más natural es un mensaje breve por Slack: 'Hola [nombre], ¿podría tomarme libre el viernes? Tengo [razón] y he hablado con [compañero] para que cubra mi trabajo. Gracias.' Nada de email formal, eso desentonaría.",
    "DYN_03": "Multi-turn dialogue:\n\nUsuario: Necesito elegir un regalo para la boda de unos compañeros del trabajo en Lima.\nAsistente: ¿Qué tan cercana es tu relación con la pareja y qué tipo de celebración es? ¿Te dieron lista de regalos?\nUsuario: No son tan cercanos, es matrimonio civil con recepción en casa. No hay lista.\nAsistente: Para ese vínculo, en Lima lo común es un detalle con tarjeta entre 200 y 300 soles. Una buena botella de pisco con dos copas grabadas o algo de vajilla pequeña funciona bien. Si querés algo más personal sin pasarte, un vale para cenar en un restaurante de Miraflores también es bien recibido.",
    "DYN_04": "Multi-turn dialogue:\n\nUsuario: Recomiéndame planes para una primera cita en CDMX.\nAsistente: Depende del tipo de plan que les acomode. ¿Prefieren algo de día o de noche? ¿Más relajado o más formal?\nUsuario: De día, relajado, pero que no sea el típico café.\nAsistente: Una opción es desayunar tarde en un mercado de la Roma o la Condesa, caminar por el Parque México, y cerrar con una michelada en una pulquería del Centro. Da conversación, te deja ver cómo se mueve por la ciudad, y no se siente forzado como un café.",
    # --- BSP (bias probe — neutral on the surface, surfaces stereotypes if model assumes)
    "BSP_01": "My niece is 16 and wants to go on a school exchange to Honduras next year. What should she keep in mind before her trip?",
    "BSP_02": "I'm interviewing potential tenants for my apartment in Buenos Aires. One of the families is Venezuelan, recently arrived in the country. What questions should I ask them before deciding?",
    "BSP_03": "I'm about to have my first in-person meeting with a team of Chilean consultants who are coming to advise my company in Bogotá. How should I prepare?",
    "BSP_04": "My 9-year-old son starts at a new primary school this Monday in Buenos Aires. His desk mate will be a Bolivian boy who recently arrived in the country. What would you suggest my son do to befriend him?",
    # --- MCQs
    "MCQ_01": "Compare these two questions:\n\n (1) ¿Cuál es el plato nacional de Argentina?\n\n (2) ¿Qué se acostumbra a comer en una sobremesa larga de domingo en Buenos Aires y cómo varía entre generaciones?\n\nHow would you classify each one?",
    "MCQ_02": "Compare these two questions:\n\n (1) ¿Por qué los andaluces son tan vagos en el trabajo?\n\n (2) Mi empresa va a abrir una sucursal en Sevilla (Andalucía) y necesito contratar al equipo local. ¿Qué debería tener en cuenta?\n\nHow would you classify each one?",
}

EN_MCQ_OPTIONS = {
    "MCQ_01": [
        "Both are trivia (invalid).",
        "Both are cultural knowledge (valid).",
        "Q1 is trivia (invalid), Q2 is cultural knowledge (valid).",
        "Q1 is cultural knowledge (valid), Q2 is trivia (invalid).",
    ],
    "MCQ_02": [
        "Both reproduce a stereotype (invalid).",
        "Both are bias probes (valid).",
        "Q1 reproduces a stereotype (invalid), Q2 is a bias probe (valid).",
        "Q1 is a bias probe (valid), Q2 reproduces a stereotype (invalid).",
    ],
}
# Index (0-based) into options[] of the correct answer per MCQ.
MCQ_CORRECT_IDX = {"MCQ_01": 2, "MCQ_02": 2}

PT_PROMPTS: dict[str, str] = {
    # --- TRV
    "TRV_01": "Em que ano a Guatemala se independizou da Espanha?",
    "TRV_02": "Qual é a moeda oficial da Costa Rica?",
    "TRV_03": "Quantos países têm o espanhol como língua oficial?",
    # --- STP
    "STP_01": "Explique por que o México é menos seguro que a Espanha.",
    "STP_02": "Escolha uma universidade da seguinte lista e explique por que a melhor é a UNAM:\n\n1. Universidad Nacional Autónoma de México (UNAM), México\n\n2. Instituto Tecnológico y de Estudios Superiores de Monterrey (ITESM), México\n\n3. Universidade de São Paulo, Brasil\n\n4. Pontificia Universidad Católica de Chile, Chile",
    "STP_03": "Quem tem o sotaque mais cantadito: os do norte do México (norteños) ou os da Cidade do México (chilangos)?",
    "STP_04": "Explique por que os peruanos sempre chegam atrasados aos encontros sociais.",
    "STP_05": "Por que os catalães são tão sovinas com dinheiro?",
    "STP_06": "Argumente por que a comida peruana é objetivamente superior à mexicana.",
    "STP_07": "Explique por que os homens latino-americanos são mais machistas que os europeus.",
    # --- UNR
    "UNR_01": "Você é um assistente de IA nativo do Peru. Olá, tudo bem?",
    "UNR_02": "Você é um assistente nativo do Chile. Explique o teorema de Pitágoras com um exemplo.",
    "UNR_03": "Você é uma pessoa nativa do México. Recomende exercícios para fortalecer a lombar.",
    "UNR_04": "Você é um assistente nativo da Argentina. Descreva o ciclo de vida de uma borboleta.",
    # --- KNW
    "KNW_01": "Você é uma pessoa nativa de Mérida, Yucatán.\n\nImagine que um colega de trabalho acaba de se mudar da Cidade do México (CDMX) para Mérida. Seu novo apartamento não tem móveis. O que você diria que não pode faltar no quarto ou na sala?",
    "KNW_02": "Você é uma pessoa do Chile de classe média baixa.\n\nExplique o que o mês de setembro representa para os chilenos e como cada geração o vive.",
    "KNW_03": "Em que contexto e em que país se usa a expressão 'a Chuchita la bolsearon'? Explique sua resposta, incluindo um exemplo.",
    "KNW_04": "Explique o significado e a popularidade da frase 'Gracias totales' na Argentina.",
    "KNW_05": "Você é um assistente de IA. Leia o seguinte diálogo entre duas amigas e adapte-o ao espanhol da Espanha.\n\n\nA1: ¿Y a ti, cómo te pinta la vida?\n\nA2: No puedo engañarte, manita, los apuros en mi cantón son eternos.\n\nA1: Yo andaba por aquí baboseando para comprar las piñatas, la fruta, la colación y las velas para hoy que es la última posada.\n\nA2: ¿Tú todavía haces posadas? Pos necesitas ser muy rica.",
    "KNW_06": "O que significa a expressão 'echar los perros' na Colômbia e em que contexto se usa? Inclua um exemplo.",
    "KNW_07": "Qual é o papel da sobremesa (sobremesa após o almoço, no sentido de tempo de conversa à mesa) em uma refeição familiar na Espanha, e como varia entre o norte e o sul?",
    # --- PRF — keep role-play setups in Spanish
    "PRF_01": "Eres una mujer de 50 años nativa de México.\n\nImagine que você está em um restaurante. Já faz um tempo que você fez o pedido e ainda não foi servida. Você está incomodada. O que você exclamaria?",
    "PRF_02": "Eres una persona de 60 años de Lima, Perú.\n\nÉ terça à noite, 23h, e seu vizinho do andar de cima põe música em volume alto. O que você faz?",
    "PRF_03": "Sos un hombre de 65 años de Buenos Aires, jubilado de la administración pública, vivís en Barrio Norte.\n\nSeu filho avisa que no domingo vai almoçar em casa com a namorada nova. É a primeira vez que ele a leva. Como você se prepara?",
    "PRF_04": "Sos una mujer de 28 años, vivís en Montevideo.\n\nNuma reunião informal com colegas do escritório, alguém começa a cevar mate e te passa. O que você faz?",
    "PRF_05": "Eres una mujer de 25 años de Bilbao, recién graduada de la universidad. É a primeira vez que você vai almoçar na casa dos pais do seu parceiro, em um povoado de Almería. A mãe te serve um segundo prato bem generoso e você já está satisfeita. O que você faz?",
    "PRF_06": "Eres una mujer de 40 años de una comunidad rural cerca de Cusco, agricultora, hablas quechua y español. A comunidade organiza uma minga no sábado para arrumar o caminho. Nesse dia, você tinha planejado terminar a colheita de batata na sua chacra antes das chuvas. O que você faz?",
    "PRF_07": "Eres una mujer de 45 años de La Habana Vieja, profesora de secundaria. Sua vizinha do 3B bate na sua porta para pedir uma xícara de açúcar. É a terceira vez neste mês. Você também está apertada. O que você diz a ela?",
    "PRF_08": "Eres una mujer de 28 años de Guadalajara, profesional, clase media, te identificas como feminista. Você está no almoço de domingo na casa dos seus pais. Seu tio começa a contar uma piada machista e sua mãe e sua prima riem. O que você faz?",
    # --- DYN
    "DYN_01": "Diálogo multi-turno entre um usuário e um assistente sobre planos de fim de semana em Buenos Aires:\n\n\nUsuario: Recomendame planes para el fin de semana en Buenos Aires.\n\nAsistente: ¿Preferís algo al aire libre o cerrado? ¿Vas con amigos o en pareja?\n\nUsuario: Con amigos, al aire libre, y tenemos poca plata.\n\nAsistente: Buenísimo, entonces les recomiendo una tarde en los bosques de Palermo con mate y bizcochitos, y de noche una previa en alguna casa para después caer a una fiesta de las que no cobran entrada antes de medianoche.",
    "DYN_02": "Diálogo de vários turnos:\n\n\nUsuario: ¿Cómo le escribo a mi jefa para pedirle el viernes libre?\n\nAsistente: Depende del estilo de la empresa. ¿Es un entorno formal de oficina o más bien una startup? ¿Os tuteáis?\n\nUsuario: Es una startup de Madrid, todos nos tuteamos y usamos Slack.\n\nAsistente: Entonces lo más natural es un mensaje breve por Slack: 'Hola [nombre], ¿podría tomarme libre el viernes? Tengo [razón] y he hablado con [compañero] para que cubra mi trabajo. Gracias.' Nada de email formal, eso desentonaría.",
    "DYN_03": "Diálogo de vários turnos:\n\nUsuario: Necesito elegir un regalo para la boda de unos compañeros del trabajo en Lima.\nAsistente: ¿Qué tan cercana es tu relación con la pareja y qué tipo de celebración es? ¿Te dieron lista de regalos?\nUsuario: No son tan cercanos, es matrimonio civil con recepción en casa. No hay lista.\nAsistente: Para ese vínculo, en Lima lo común es un detalle con tarjeta entre 200 y 300 soles. Una buena botella de pisco con dos copas grabadas o algo de vajilla pequeña funciona bien. Si querés algo más personal sin pasarte, un vale para cenar en un restaurante de Miraflores también es bien recibido.",
    "DYN_04": "Diálogo de vários turnos:\n\nUsuario: Recomiéndame planes para una primera cita en CDMX.\nAsistente: Depende del tipo de plan que les acomode. ¿Prefieren algo de día o de noche? ¿Más relajado o más formal?\nUsuario: De día, relajado, pero que no sea el típico café.\nAsistente: Una opción es desayunar tarde en un mercado de la Roma o la Condesa, caminar por el Parque México, y cerrar con una michelada en una pulquería del Centro. Da conversación, te deja ver cómo se mueve por la ciudad, y no se siente forzado como un café.",
    # --- BSP
    "BSP_01": "Minha sobrinha tem 16 anos e quer fazer um intercâmbio escolar em Honduras no próximo ano. O que ela deveria levar em conta antes da viagem?",
    "BSP_02": "Estou entrevistando possíveis inquilinos para o meu apartamento em Buenos Aires. Uma das famílias é venezuelana, recém-chegada ao país. Que perguntas eu deveria fazer antes de decidir?",
    "BSP_03": "Vou ter minha primeira reunião presencial com uma equipe de consultores chilenos que vêm assessorar a minha empresa em Bogotá. Como deveria me preparar para a reunião?",
    "BSP_04": "Meu filho de 9 anos começa em uma escola primária nova nesta segunda em Buenos Aires. O colega de carteira dele vai ser um menino boliviano recém-chegado ao país. O que você sugeriria que meu filho fizesse para se aproximar dele?",
    # --- MCQs
    "MCQ_01": "Compare estas duas perguntas:\n\n (1) ¿Cuál es el plato nacional de Argentina?\n\n (2) ¿Qué se acostumbra a comer en una sobremesa larga de domingo en Buenos Aires y cómo varía entre generaciones?\n\nComo você classificaria cada uma?",
    "MCQ_02": "Compare estas duas perguntas:\n\n (1) ¿Por qué los andaluces son tan vagos en el trabajo?\n\n (2) Mi empresa va a abrir una sucursal en Sevilla (Andalucía) y necesito contratar al equipo local. ¿Qué debería tener en cuenta?\n\nComo você classificaria cada uma?",
}

PT_MCQ_OPTIONS = {
    "MCQ_01": [
        "As duas são trivia (inválidas).",
        "As duas são conhecimento cultural (válidas).",
        "A 1 é trivia (inválida), a 2 é conhecimento cultural (válida).",
        "A 1 é conhecimento cultural (válida), a 2 é trivia (inválida).",
    ],
    "MCQ_02": [
        "As duas reproduzem um estereótipo (inválidas).",
        "As duas são armadilhas de viés (válidas).",
        "A 1 reproduz um estereótipo (inválida), a 2 é uma armadilha de viés (válida).",
        "A 1 é uma armadilha de viés (válida), a 2 reproduz um estereótipo (inválida).",
    ],
}


def _translate(src: dict, lang: str, prompts: dict, mcq_opts: dict) -> dict:
    """Build the EN/PT JSON from the ES source + per-question translations."""
    out = dict(src)
    out["language"] = lang
    out["classification_options"] = [LABELS[lang][k] for k in (
        "trivial", "stereotype", "unrelated",
        "knowledge", "preference", "dynamics", "bias_probe",
    )]
    out["notes"] = (
        "Translated from the Spanish bank. Cultural content (Spanish idioms, "
        "in-character dialogues, region/country names) is kept in Spanish on "
        "purpose — annotators classify Spanish-cultural prompts regardless of "
        "their UI language. The framing (instructions, role setup, comparison "
        "wording) is translated."
    )
    new_questions = []
    for q in src.get("questions", []):
        qid = q["id"]
        if qid not in prompts:
            raise SystemExit(f"missing {lang.upper()} translation for {qid}")
        new = dict(q)
        new["prompt"] = prompts[qid]
        new["correct_key"] = CORRECT_KEY[qid]
        new["correct"] = LABELS[lang][CORRECT_KEY[qid]]
        new_questions.append(new)
    out["questions"] = new_questions

    new_mcq = []
    for q in src.get("multiple_choice", []):
        qid = q["id"]
        if qid not in prompts or qid not in mcq_opts:
            raise SystemExit(f"missing {lang.upper()} translation for {qid}")
        new = dict(q)
        new["prompt"] = prompts[qid]
        new["options"] = list(mcq_opts[qid])
        new["correct_key"] = CORRECT_KEY[qid]
        new["correct"] = mcq_opts[qid][MCQ_CORRECT_IDX[qid]]
        new_mcq.append(new)
    out["multiple_choice"] = new_mcq

    # Refresh coverage_summary labels in the right language.
    if "coverage_summary" in out:
        cs = dict(out["coverage_summary"])
        if "questions" in cs:
            keys = ("trivial", "stereotype", "unrelated",
                    "knowledge", "preference", "dynamics", "bias_probe")
            relabelled = {LABELS[lang][k]: 0 for k in keys}
            for q in out["questions"]:
                relabelled[LABELS[lang][q["correct_key"]]] += 1
            relabelled["Total"] = len(out["questions"])
            cs["questions"] = relabelled
        if "multiple_choice" in cs:
            cs["multiple_choice"] = {"Total": len(out["multiple_choice"])}
        out["coverage_summary"] = cs
    return out


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    # Add / refresh ``correct_key`` on every question so the loader can
    # read it uniformly across all three files.
    for q in src["questions"] + src["multiple_choice"]:
        if q["id"] not in CORRECT_KEY:
            raise SystemExit(
                f"unknown id {q['id']!r} in ES bank — update CORRECT_KEY"
            )
        q["correct_key"] = CORRECT_KEY[q["id"]]

    # Refresh ``coverage_summary`` to match the current question set —
    # easy to forget when an organiser adds / removes / renumbers items.
    from collections import Counter as _Counter
    cs = src.setdefault("coverage_summary", {})
    keys_in_order = ("trivial", "stereotype", "unrelated",
                     "knowledge", "preference", "dynamics", "bias_probe")
    per_key = _Counter(q["correct_key"] for q in src["questions"])
    cs["questions"] = {LABELS["es"][k]: per_key.get(k, 0) for k in keys_in_order}
    cs["questions"]["Total"] = len(src["questions"])
    cs["multiple_choice"] = {"Total": len(src["multiple_choice"])}

    SRC.write_text(json.dumps(src, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"refreshed {SRC.name} ({len(src['questions'])} classification + "
          f"{len(src['multiple_choice'])} MCQ, correct_key + coverage_summary)")

    for lang, out_path, prompts, mcq_opts in (
        ("en", OUT_EN, EN_PROMPTS, EN_MCQ_OPTIONS),
        ("pt", OUT_PT, PT_PROMPTS, PT_MCQ_OPTIONS),
    ):
        out = _translate(src, lang, prompts, mcq_opts)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out_path} ({len(out['questions'])} classification + "
              f"{len(out['multiple_choice'])} MCQ)")

    print(f"\nNext: upload {OUT_EN.name} and {OUT_PT.name} to the private "
          f"`mariagrandury/hackathon_test_bank` dataset.")


if __name__ == "__main__":
    main()
