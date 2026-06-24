# Claude annotation analysis

Joined `mariagrandury/cultural_preferences_claude` against `mariagrandury/cultural_preferences` — 2920 prompts.

## 1. Coverage

- prompts joined: **2920**
- `validation_choice`: 2920 (100%)
- `vote_choice`: 1120 (38%)
- `region`: 1272 (44%)
- `cultural_topic`: 2920 (100%)
- `d1_dimension`: 2920 (100%)
- `d2_topic`: 2920 (100%)
- `d3_register`: 2920 (100%)
- `d4_complexity`: 2920 (100%)
- `d5_multilingual`: 2920 (100%)
- `d6_anchoring`: 2920 (100%)
- prompts with ≥1 human validation slot filled: **1463**
- prompts with ≥1 human vote: **0**

## 2. Distributions

### `validation_choice`  (n=2920)

| value | count | % |
|---|---:|---:|
| knowledge | 1772 | 60.7 |
| preference | 491 | 16.8 |
| dynamics | 397 | 13.6 |
| bias_probe | 148 | 5.1 |
| trivial | 91 | 3.1 |
| unrelated | 13 | 0.4 |
| stereotype | 8 | 0.3 |

### `cultural_topic`  (n=2920)

| value | count | % |
|---|---:|---:|
| language_and_dialect | 969 | 33.2 |
| social_norms_and_etiquette | 387 | 13.3 |
| work_and_occupations | 239 | 8.2 |
| food_and_drink | 233 | 8.0 |
| values_and_opinions | 233 | 8.0 |
| geography_and_local_life | 207 | 7.1 |
| festivals_and_celebrations | 138 | 4.7 |
| sports_and_leisure | 94 | 3.2 |
| family_and_kinship | 94 | 3.2 |
| arts_music_dance | 91 | 3.1 |
| history_and_heritage | 87 | 3.0 |
| traditions_and_customs | 60 | 2.1 |
| religion_and_beliefs | 53 | 1.8 |
| other | 27 | 0.9 |
| clothing_and_appearance | 8 | 0.3 |

### `vote_choice`  (n=1120)

| value | count | % |
|---|---:|---:|
| b | 471 | 42.1 |
| a | 415 | 37.1 |
| both | 214 | 19.1 |
| none | 20 | 1.8 |

### `d1_dimension`  (n=2920)

| value | count | % |
|---|---:|---:|
| Conocimiento | 1693 | 58.0 |
| Preferencia | 967 | 33.1 |
| Trampa de sesgo | 145 | 5.0 |
| Dinámica | 91 | 3.1 |
| NONE | 24 | 0.8 |

### `d2_topic`  (n=2920)

| value | count | % |
|---|---:|---:|
| Ideacional - Conceptos | 781 | 26.7 |
| Ideacional - Conocimiento | 553 | 18.9 |
| Social - Contexto | 464 | 15.9 |
| Lingüística - Dialectos | 220 | 7.5 |
| Social - Demografía | 176 | 6.0 |
| Ideacional - Normas y morales | 151 | 5.2 |
| Ideacional - Artefactos | 147 | 5.0 |
| Social - Intención comunicativa | 143 | 4.9 |
| Social - Relaciones | 128 | 4.4 |
| Ideacional - Valores | 83 | 2.8 |
| Lingüística - Estilos/registros/géneros | 74 | 2.5 |

### `d3_register`  (n=2920)

| value | count | % |
|---|---:|---:|
| Neutro | 2383 | 81.6 |
| Informal | 390 | 13.4 |
| Formal | 117 | 4.0 |
| Mixto | 30 | 1.0 |

### `d4_complexity`  (n=2920)

| value | count | % |
|---|---:|---:|
| Media | 2186 | 74.9 |
| Baja | 550 | 18.8 |
| Alta | 184 | 6.3 |

### `d5_multilingual`  (n=2920)

| value | count | % |
|---|---:|---:|
| Monolingüe | 2919 | 100.0 |
| Code-switching | 1 | 0.0 |

### `d6_anchoring`  (n=2920)

| value | count | % |
|---|---:|---:|
| Alto | 2273 | 77.8 |
| Medio | 499 | 17.1 |
| Bajo | 148 | 5.1 |

### `region` (top 15 of 1272)

| region | count |
|---|---:|
| La Habana | 47 |
| Madrid | 34 |
| Santiago | 26 |
| Lima | 26 |
| Quito | 23 |
| Camagüey | 21 |
| Matanzas | 20 |
| Santa Clara | 20 |
| Guayaquil | 20 |
| Holguín | 19 |
| Santiago de Cuba | 18 |
| Cienfuegos | 16 |
| Buenos Aires | 15 |
| Centro Habana | 13 |
| Sevilla | 13 |

## 3. Claude-vs-human agreement

- **validation verdict** (accept/reject), ≥1 human slot: 1246/1463 (**85%**)
- validation verdict, all 3 human slots filled: 18/25 (72%)
- **fine bucket** (Claude vs human majority): 712/1463 (49%)
- **A/B vote** (Claude vs human majority): 0/0 (no human votes yet)

Validation verdict confusion (rows=human, cols=Claude):

```
claude_verdict  accept  reject
human_verdict                 
accept            1232      22
reject             195      14
```

## 4. Association with country (culture)

Cramér's V (0=independent, 1=fully determined) of each annotation vs `country`:

| annotation | Cramér's V | n |
|---|---:|---:|
| `d3_register` | 0.357 | 2920 |
| `validation_choice` | 0.345 | 2920 |
| `d4_complexity` | 0.323 | 2920 |
| `d2_topic` | 0.296 | 2920 |
| `d6_anchoring` | 0.276 | 2920 |
| `cultural_topic` | 0.258 | 2920 |
| `d1_dimension` | 0.253 | 2920 |
| `vote_choice` | 0.194 | 1120 |
| `d5_multilingual` | 0.043 | 2920 |


**Most distinctive cultural_topic per country (lift ≥1.5, n≥20):**

| country | cultural_topic | rate | lift |
|---|---|---:|---:|
| ar | clothing_and_appearance | 1% | 2.4× |
| ca | clothing_and_appearance | 4% | 14.3× |
| cl | social_norms_and_etiquette | 33% | 2.5× |
| co | history_and_heritage | 13% | 4.3× |
| cu | language_and_dialect | 61% | 1.9× |
| ec | clothing_and_appearance | 1% | 4.8× |
| es | other | 4% | 3.8× |
| mx | other | 5% | 5.4× |
| ni | values_and_opinions | 29% | 3.6× |
| pe | sports_and_leisure | 12% | 3.7× |
| py | history_and_heritage | 17% | 5.7× |


**Most distinctive D1 per country (lift ≥1.3, n≥20):**

| country | d1_dimension | rate | lift |
|---|---|---:|---:|
| ar | Dinámica | 13% | 4.3× |
| ca | Trampa de sesgo | 24% | 4.7× |
| cl | Preferencia | 57% | 1.7× |
| co | Conocimiento | 83% | 1.4× |
| cu | Trampa de sesgo | 8% | 1.6× |
| ec | Preferencia | 49% | 1.5× |
| es | NONE | 5% | 6.2× |
| mx | Conocimiento | 100% | 1.7× |
| pe | Preferencia | 57% | 1.7× |
| py | Conocimiento | 82% | 1.4× |

## 5. Cross-dimension associations (Cramér's V)

Pairwise association between Claude's categorical annotations:

| pair | Cramér's V | n |
|---|---:|---:|
| `validation_choice` × `d1_dimension` | 0.618 | 2920 |
| `d1_dimension` × `d2_topic` | 0.557 | 2920 |
| `validation_choice` × `d2_topic` | 0.488 | 2920 |
| `d1_dimension` × `d6_anchoring` | 0.451 | 2920 |
| `cultural_topic` × `d2_topic` | 0.439 | 2920 |
| `d2_topic` × `d6_anchoring` | 0.402 | 2920 |
| `validation_choice` × `cultural_topic` | 0.396 | 2920 |
| `cultural_topic` × `d6_anchoring` | 0.391 | 2920 |
| `validation_choice` × `d6_anchoring` | 0.379 | 2920 |
| `cultural_topic` × `d1_dimension` | 0.367 | 2920 |
| `d3_register` × `d4_complexity` | 0.331 | 2920 |
| `d1_dimension` × `d3_register` | 0.291 | 2920 |
| `d2_topic` × `d3_register` | 0.277 | 2920 |
| `validation_choice` × `d3_register` | 0.274 | 2920 |
| `d2_topic` × `d4_complexity` | 0.217 | 2920 |
| `d1_dimension` × `d4_complexity` | 0.187 | 2920 |
| `validation_choice` × `d4_complexity` | 0.182 | 2920 |
| `cultural_topic` × `d4_complexity` | 0.175 | 2920 |
| `cultural_topic` × `d3_register` | 0.162 | 2920 |
| `vote_choice` × `d2_topic` | 0.143 | 1120 |

**`validation_choice` (rows) × `d1_dimension` (cols)** — two independent passes at the cultural dimension:

```
d1_dimension       Conocimiento  Dinámica  NONE  Preferencia  Trampa de sesgo
validation_choice                                                            
bias_probe                    1         0     0           27              120
dynamics                     18        89     0          278               12
knowledge                  1546         0     8          217                1
preference                   45         2     6          436                2
stereotype                    0         0     0            0                8
trivial                      82         0     5            3                1
unrelated                     1         0     5            6                1
```

## 6. Agreement breakdown (where humans validated)

**verdict agreement by `country`** (groups with n≥3):

| country | agreement | n |
|---|---:|---:|
| mx | 0% | 8 |
| pe | 60% | 126 |
| co | 77% | 48 |
| ar | 80% | 99 |
| py | 83% | 64 |
| es | 84% | 345 |
| cu | 88% | 420 |
| ec | 94% | 31 |
| cl | 97% | 313 |
| ca | 100% | 5 |
| ni | 100% | 4 |

**verdict agreement by `cultural_topic`** (groups with n≥3):

| cultural_topic | agreement | n |
|---|---:|---:|
| sports_and_leisure | 64% | 55 |
| other | 69% | 16 |
| work_and_occupations | 73% | 111 |
| history_and_heritage | 78% | 37 |
| family_and_kinship | 79% | 48 |
| language_and_dialect | 82% | 482 |
| values_and_opinions | 87% | 123 |
| festivals_and_celebrations | 89% | 57 |
| social_norms_and_etiquette | 90% | 220 |
| food_and_drink | 94% | 129 |
| arts_music_dance | 94% | 34 |
| geography_and_local_life | 95% | 105 |
| traditions_and_customs | 96% | 27 |
| religion_and_beliefs | 100% | 17 |

**verdict agreement by `d1_dimension`** (groups with n≥3):

| d1_dimension | agreement | n |
|---|---:|---:|
| Trampa de sesgo | 78% | 58 |
| Preferencia | 81% | 540 |
| NONE | 81% | 21 |
| Dinámica | 88% | 43 |
| Conocimiento | 89% | 801 |

## 7. Auto-surfaced findings

- Claude rejects **112/2920 (3.8%)** of prompts.
- Topics most often rejected: `other` 63%, `history_and_heritage` 26%, `sports_and_leisure` 9%, `language_and_dialect` 4%, `geography_and_local_life` 4%.
- Blind A/B votes: b=471, a=415, both=214, none=20 (A/B balance 415/471).
- `d6_anchoring`: dominated by **Alto** (78%).
- `d4_complexity`: dominated by **Media** (75%).
- `d3_register`: dominated by **Neutro** (82%).
- `d5_multilingual`: dominated by **Monolingüe** (100%).
- D1=`NONE` on 24 prompts; of those Claude validation-rejects 10 (42%) — consistency check between the two passes.

**217 verdict disagreements with humans** (ids): 6, 90, 96, 105, 108, 139, 155, 158, 164, 165, 173, 175, 184, 185, 188, 196, 197, 199, 201, 208, 209, 211, 217, 220, 222.

