"""Analyse the Claude annotations in ``cultural_preferences_claude``.

Joins Claude's labels (validation bucket, blind A/B vote, region, cultural_topic,
and the six analysis.md dimensions ``d1_dimension``..``d6_anchoring``) against the
human ``cultural_preferences`` dataset and reports:

  1. Coverage of every Claude field.
  2. Per-field distributions.
  3. Claude-vs-human agreement: validation verdict (accept/reject), fine bucket,
     and blind A/B votes — wherever humans have caught up.
  4. How each categorical annotation associates with country (Cramér's V), plus
     the most over-represented topic/dimension per country.
  5. Cross-dimension associations (Cramér's V matrix) — e.g. does Claude's
     validation bucket line up with analysis.md's D1, does cultural_topic line up
     with D2.
  6. Agreement broken down by country / cultural_topic / D1 (where humans
     validated).
  7. Auto-surfaced "interesting" findings.

Pure read-only: never writes to any dataset. Emits a Markdown report and prints
highlights.

Usage:
    python analyze_claude_annotations.py [--out data/analysis_claude_annotations.md]
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import annotate_claude as ac
import data

# Claude annotation fields we treat as categorical for distributions/association.
CLAUDE_CATS = (
    "validation_choice",
    "cultural_topic",
    "vote_choice",
    "d1_dimension",
    "d2_topic",
    "d3_register",
    "d4_complexity",
    "d5_multilingual",
    "d6_anchoring",
)


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def _chi2(ct: pd.DataFrame) -> float:
    obs = ct.values.astype(float)
    row = obs.sum(1, keepdims=True)
    col = obs.sum(0, keepdims=True)
    tot = obs.sum()
    if tot == 0:
        return float("nan")
    exp = row @ col / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (obs - exp) ** 2 / exp
    return float(np.nansum(terms))


def cramers_v(s1: pd.Series, s2: pd.Series) -> tuple[float, int]:
    """Cramér's V association between two categorical series (0..1), plus the
    number of paired non-empty observations it was computed on."""
    m = (s1.astype(str).str.strip() != "") & (s2.astype(str).str.strip() != "")
    a, b = s1[m], s2[m]
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan"), n
    chi2 = _chi2(ct)
    denom = n * (min(ct.shape) - 1)
    return (float(np.sqrt(chi2 / denom)) if denom else float("nan")), n


def _nonempty(s: pd.Series) -> pd.Series:
    return s[s.astype(str).str.strip() != ""]


def dist_table(s: pd.Series, top: int | None = None) -> list[tuple[str, int, float]]:
    v = _nonempty(s).value_counts()
    if top:
        v = v.head(top)
    tot = int(v.sum()) or 1
    return [(str(k), int(c), 100 * c / tot) for k, c in v.items()]


# --------------------------------------------------------------------------- #
# human-side derived columns
# --------------------------------------------------------------------------- #
def _human_validation(row) -> tuple[int, str | None, str | None]:
    """Return (n_filled_slots, verdict, majority_bucket) for a prompt row.
    verdict: 'reject' if any filled slot is a reject bucket, else 'accept', else
    None when no slot is filled. majority_bucket: modal filled choice or None."""
    choices = [
        row[c]["choice"]
        for c in data.VALIDATION_COLS
        if isinstance(row[c], dict) and row[c].get("username")
    ]
    if not choices:
        return 0, None, None
    verdict = "accept" if all(c in data.ACCEPT_VALIDATION_CHOICES for c in choices) else "reject"
    majority = Counter(choices).most_common(1)[0][0]
    return len(choices), verdict, majority


def _human_vote(row) -> tuple[int, str | None]:
    choices = [
        row[c]["choice"]
        for c in data.VOTE_COLS
        if isinstance(row[c], dict) and row[c].get("username") and row[c].get("choice")
    ]
    if not choices:
        return 0, None
    return len(choices), Counter(choices).most_common(1)[0][0]


def build_joined() -> pd.DataFrame:
    prompts = data.load_prompts_df()
    claude = ac.load_claude_df()
    # Claude fields, prefixed where they'd collide.
    csub = claude.set_index("id")
    keep = [c for c in claude.columns if c != "id"]
    df = prompts.set_index("id").join(csub[keep], how="left", rsuffix="_claude")
    df = df.reset_index()
    # human-derived
    hv = df.apply(_human_validation, axis=1, result_type="expand")
    df["human_n_val"], df["human_verdict"], df["human_bucket"] = hv[0], hv[1], hv[2]
    vt = df.apply(_human_vote, axis=1, result_type="expand")
    df["human_n_vote"], df["human_vote"] = vt[0], vt[1]
    # claude verdict (accept/reject) from its single validation bucket
    df["claude_verdict"] = df["validation_choice"].map(
        lambda c: "accept" if c in data.ACCEPT_VALIDATION_CHOICES else ("reject" if str(c).strip() else "")
    )
    return df


# --------------------------------------------------------------------------- #
# report sections (each returns a list of markdown lines)
# --------------------------------------------------------------------------- #
def sec_coverage(df: pd.DataFrame) -> list[str]:
    n = len(df)
    out = ["## 1. Coverage", "", f"- prompts joined: **{n}**"]
    for f in ("validation_choice", "vote_choice", "region", "cultural_topic", *ac.DIM_FIELDS):
        c = int((df[f].astype(str).str.strip() != "").sum())
        out.append(f"- `{f}`: {c} ({100*c/n:.0f}%)")
    out.append(f"- prompts with ≥1 human validation slot filled: **{int((df['human_n_val']>0).sum())}**")
    out.append(f"- prompts with ≥1 human vote: **{int((df['human_n_vote']>0).sum())}**")
    return out + [""]


def sec_distributions(df: pd.DataFrame) -> list[str]:
    out = ["## 2. Distributions", ""]
    for f in CLAUDE_CATS:
        rows = dist_table(df[f])
        if not rows:
            continue
        out.append(f"### `{f}`  (n={sum(r[1] for r in rows)})")
        out.append("")
        out.append("| value | count | % |")
        out.append("|---|---:|---:|")
        for k, c, p in rows:
            out.append(f"| {k} | {c} | {p:.1f} |")
        out.append("")
    # region: just the head, it's high-cardinality
    rt = dist_table(df["region"], top=15)
    if rt:
        out.append(f"### `region` (top 15 of {int((df['region'].astype(str).str.strip()!='').sum())})")
        out.append("")
        out.append("| region | count |")
        out.append("|---|---:|")
        for k, c, _ in rt:
            out.append(f"| {k} | {c} |")
        out.append("")
    return out


def _agreement(a: pd.Series, b: pd.Series) -> tuple[int, int]:
    m = a.notna() & b.notna() & (a.astype(str).str.strip() != "") & (b.astype(str).str.strip() != "")
    return int((a[m] == b[m]).sum()), int(m.sum())


def sec_agreement(df: pd.DataFrame) -> list[str]:
    out = ["## 3. Claude-vs-human agreement", ""]
    # validation verdict (accept/reject), where humans filled ≥1 slot
    sub = df[(df["human_n_val"] > 0) & (df["claude_verdict"] != "")]
    ag, n = _agreement(sub["claude_verdict"], sub["human_verdict"])
    out.append(f"- **validation verdict** (accept/reject), ≥1 human slot: {ag}/{n}"
               + (f" (**{100*ag/n:.0f}%**)" if n else " (no overlap yet)"))
    # strict: all 3 human slots filled
    strict = df[(df["human_n_val"] >= 3) & (df["claude_verdict"] != "")]
    ag2, n2 = _agreement(strict["claude_verdict"], strict["human_verdict"])
    out.append(f"- validation verdict, all 3 human slots filled: {ag2}/{n2}"
               + (f" ({100*ag2/n2:.0f}%)" if n2 else " (none yet)"))
    # fine bucket vs human majority bucket
    fb = df[(df["human_n_val"] > 0) & (df["validation_choice"].astype(str).str.strip() != "")]
    ag3, n3 = _agreement(fb["validation_choice"], fb["human_bucket"])
    out.append(f"- **fine bucket** (Claude vs human majority): {ag3}/{n3}"
               + (f" ({100*ag3/n3:.0f}%)" if n3 else " (none yet)"))
    # votes
    vv = df[(df["human_n_vote"] > 0) & (df["vote_choice"].astype(str).str.strip() != "")]
    ag4, n4 = _agreement(vv["vote_choice"], vv["human_vote"])
    out.append(f"- **A/B vote** (Claude vs human majority): {ag4}/{n4}"
               + (f" ({100*ag4/n4:.0f}%)" if n4 else " (no human votes yet)"))
    out.append("")
    # confusion of validation verdict if any overlap
    if n:
        ct = pd.crosstab(sub["human_verdict"], sub["claude_verdict"])
        out.append("Validation verdict confusion (rows=human, cols=Claude):")
        out.append("")
        out.append("```")
        out.append(ct.to_string())
        out.append("```")
        out.append("")
    return out


def sec_country(df: pd.DataFrame) -> list[str]:
    out = ["## 4. Association with country (culture)", "",
           "Cramér's V (0=independent, 1=fully determined) of each annotation vs `country`:", ""]
    out.append("| annotation | Cramér's V | n |")
    out.append("|---|---:|---:|")
    rows = []
    for f in CLAUDE_CATS:
        v, n = cramers_v(df[f], df["country"])
        if n:
            rows.append((f, v, n))
    for f, v, n in sorted(rows, key=lambda r: (-(r[1] if r[1]==r[1] else -1))):
        out.append(f"| `{f}` | {v:.3f} | {n} |")
    out.append("")
    # most over-represented cultural_topic per country (lift vs global)
    out += _lift_table(df, "cultural_topic", "country", "Most distinctive cultural_topic per country (lift ≥1.5, n≥20)")
    out += _lift_table(df, "d1_dimension", "country", "Most distinctive D1 per country (lift ≥1.3, n≥20)", min_lift=1.3)
    return out


def _lift_table(df, field, by, title, min_lift=1.5, min_n=20) -> list[str]:
    sub = df[(df[field].astype(str).str.strip() != "") & (df[by].astype(str).str.strip() != "")]
    if sub.empty:
        return []
    global_rate = _nonempty(sub[field]).value_counts(normalize=True)
    out = ["", f"**{title}:**", ""]
    out.append(f"| {by} | {field} | rate | lift |")
    out.append("|---|---|---:|---:|")
    any_row = False
    for c, grp in sub.groupby(by):
        if len(grp) < min_n:
            continue
        local = grp[field].value_counts(normalize=True)
        best = None
        for val, rate in local.items():
            lift = rate / global_rate.get(val, 1e-9)
            if lift >= min_lift and (best is None or lift > best[2]):
                best = (val, rate, lift)
        if best:
            any_row = True
            out.append(f"| {c} | {best[0]} | {best[1]*100:.0f}% | {best[2]:.1f}× |")
    return out + [""] if any_row else []


def sec_cross(df: pd.DataFrame) -> list[str]:
    out = ["## 5. Cross-dimension associations (Cramér's V)", "",
           "Pairwise association between Claude's categorical annotations:", ""]
    fields = [f for f in CLAUDE_CATS if (df[f].astype(str).str.strip() != "").any()]
    out.append("| pair | Cramér's V | n |")
    out.append("|---|---:|---:|")
    pairs = []
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            v, n = cramers_v(df[fields[i]], df[fields[j]])
            if n:
                pairs.append((fields[i], fields[j], v, n))
    for a, b, v, n in sorted(pairs, key=lambda r: -(r[2] if r[2] == r[2] else -1))[:20]:
        out.append(f"| `{a}` × `{b}` | {v:.3f} | {n} |")
    out.append("")
    # spotlight: Claude validation bucket vs analysis.md D1 (should align)
    sub = df[(df["validation_choice"].astype(str).str.strip() != "") & (df["d1_dimension"].astype(str).str.strip() != "")]
    if not sub.empty:
        ct = pd.crosstab(sub["validation_choice"], sub["d1_dimension"])
        out.append("**`validation_choice` (rows) × `d1_dimension` (cols)** — two independent passes at the cultural dimension:")
        out.append("")
        out.append("```")
        out.append(ct.to_string())
        out.append("```")
        out.append("")
    return out


def sec_agreement_breakdown(df: pd.DataFrame) -> list[str]:
    out = ["## 6. Agreement breakdown (where humans validated)", ""]
    sub = df[(df["human_n_val"] > 0) & (df["claude_verdict"] != "")].copy()
    if sub.empty:
        return out + ["_No human validations yet._", ""]
    sub["agree"] = (sub["claude_verdict"] == sub["human_verdict"]).astype(int)
    for by in ("country", "cultural_topic", "d1_dimension"):
        g = sub[sub[by].astype(str).str.strip() != ""].groupby(by)["agree"].agg(["mean", "count"])
        g = g[g["count"] >= 3].sort_values("mean")
        if g.empty:
            continue
        out.append(f"**verdict agreement by `{by}`** (groups with n≥3):")
        out.append("")
        out.append(f"| {by} | agreement | n |")
        out.append("|---|---:|---:|")
        for k, r in g.iterrows():
            out.append(f"| {k} | {r['mean']*100:.0f}% | {int(r['count'])} |")
        out.append("")
    return out


def sec_interesting(df: pd.DataFrame) -> list[str]:
    out = ["## 7. Auto-surfaced findings", ""]
    n = len(df)
    # reject rate overall + by topic
    rej = df["claude_verdict"] == "reject"
    out.append(f"- Claude rejects **{int(rej.sum())}/{n} ({100*rej.mean():.1f}%)** of prompts.")
    tsub = df[df["cultural_topic"].astype(str).str.strip() != ""].assign(_rej=rej.astype(int))
    topic_rej = tsub.groupby("cultural_topic")["_rej"].mean().sort_values(ascending=False)
    topic_rej = topic_rej[topic_rej > 0].head(5)
    if not topic_rej.empty:
        out.append("- Topics most often rejected: " + ", ".join(f"`{k}` {v*100:.0f}%" for k, v in topic_rej.items()) + ".")
    # vote balance
    vc = _nonempty(df["vote_choice"]).value_counts()
    if not vc.empty:
        out.append(f"- Blind A/B votes: " + ", ".join(f"{k}={v}" for k, v in vc.items())
                   + f" (A/B balance {vc.get('a',0)}/{vc.get('b',0)}).")
    # d6 anchoring vs d4 complexity quick read
    for f in ("d6_anchoring", "d4_complexity", "d3_register", "d5_multilingual"):
        d = _nonempty(df[f]).value_counts(normalize=True)
        if not d.empty:
            top = d.index[0]
            out.append(f"- `{f}`: dominated by **{top}** ({d.iloc[0]*100:.0f}%).")
    # NONE / not-culturally-grounded signal: D1=NONE vs Claude unrelated reject
    none_d1 = df["d1_dimension"] == "NONE"
    if none_d1.any():
        also_rej = df.loc[none_d1, "claude_verdict"] == "reject"
        out.append(f"- D1=`NONE` on {int(none_d1.sum())} prompts; of those Claude validation-rejects {int(also_rej.sum())} "
                   f"({100*also_rej.mean():.0f}%) — consistency check between the two passes.")
    # disagreement examples
    sub = df[(df["human_n_val"] > 0) & (df["claude_verdict"] != "") & (df["human_verdict"].notna())]
    disagree = sub[sub["claude_verdict"] != sub["human_verdict"]]
    if not disagree.empty:
        out.append("")
        out.append(f"**{len(disagree)} verdict disagreements with humans** (ids): "
                   + ", ".join(str(int(i)) for i in disagree["id"].head(25)) + ".")
    return out + [""]


def build_report(df: pd.DataFrame) -> str:
    parts = ["# Claude annotation analysis", "",
             f"Joined `{ac.CLAUDE_REPO}` against `{data.PROMPTS_REPO}` — {len(df)} prompts.", ""]
    for sec in (sec_coverage, sec_distributions, sec_agreement, sec_country,
                sec_cross, sec_agreement_breakdown, sec_interesting):
        parts += sec(df)
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/analysis_claude_annotations.md")
    args = ap.parse_args()
    df = build_joined()
    report = build_report(df)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    # print the highlights (coverage + agreement + findings) to stdout
    for sec in (sec_coverage, sec_agreement, sec_interesting):
        print("\n".join(sec(df)))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
