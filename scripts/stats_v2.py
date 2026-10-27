#!/usr/bin/env python3
"""
Paired statistics for the JBHI manuscript, computed from the per-case predictions.

Every contrast pairs two configurations case by case inside a seed (all methods share the
split within a seed) and pools the paired differences across seeds. For each contrast:
mean difference, bootstrap 95% CI (10,000 resamples), a Wilcoxon signed-rank p-value
(McNemar's exact test for classification accuracy), and a Holm-adjusted p within its
family. Families are the tables they feed.

  python scripts/stats_v2.py                    # writes results/stats_v2.json + tables/paired_*.tex

Kinds:  dice     per-case Dice,  A - B
        abserr   per-case |pred - gt| in years, A - B (negative favours A)
        const    per-case |pred - gt| of A minus |mean(train ages) - gt| (A vs constant predictor)
        mcnemar  per-case correctness at score > 0.5, A vs B
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import TASK_CONFIGS  # noqa: E402

OUT = ROOT / "outputs" / "benchmark_v2"
SEEDS = (42, 43, 44, 45, 46)
RNG = np.random.default_rng(0)


def per_case(n, cfg, seed, task):
    f = OUT / n / cfg / f"seed{seed}" / f"task{task}_per_case.json"
    if not f.exists():
        return None
    return {os.path.basename(c["case"]): c for c in json.load(open(f))}


def train_mean_label(task, seed, n_shot=32):
    sp = json.load(open(ROOT / "data/splits_v3" / f"{TASK_CONFIGS[task]['name']}_n{n_shot}_seed{seed}.json"))["splits"]
    vals = [float(open(p + ".txt").read().split()[0]) for p in sp["train"]]
    return float(np.mean(vals))


def paired_values(kind, cfgA, cfgB, task, n="n32", seeds=SEEDS):
    """Return (diffs or (b, c) counts, n_pairs, n_seeds_used)."""
    diffs, b, c, used = [], 0, 0, 0
    for s in seeds:
        A = per_case(n, cfgA, s, task)
        B = per_case(n, cfgB, s, task) if cfgB != "const" else None
        if A is None or (cfgB != "const" and B is None):
            continue
        used += 1
        if kind == "const":
            m = train_mean_label(task, s)
            for k, a in A.items():
                diffs.append(abs(a["pred"] - a["gt"]) - abs(m - a["gt"]))
            continue
        for k in A.keys() & B.keys():
            a, bb = A[k], B[k]
            if kind == "dice":
                diffs.append(a["dice"] - bb["dice"])
            elif kind == "abserr":
                diffs.append(abs(a["pred"] - a["gt"]) - abs(bb["pred"] - bb["gt"]))
            elif kind == "mcnemar":
                ca, cb = int((a["score"] > 0.5) == bool(a["gt"])), int((bb["score"] > 0.5) == bool(bb["gt"]))
                b += int(ca and not cb)
                c += int(cb and not ca)
                diffs.append(ca - cb)
    return np.asarray(diffs, dtype=float), b, c, used


def summarise(kind, d, b, c):
    if d.size == 0:
        return None
    mean = float(d.mean())
    boots = np.array([RNG.choice(d, d.size, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if kind == "mcnemar":
        k = min(b, c)
        p = float(min(1.0, 2 * stats.binom.cdf(k, b + c, 0.5))) if (b + c) > 0 else 1.0
        extra = {"discordant_A_only": b, "discordant_B_only": c}
    else:
        nz = d[d != 0]
        p = float(stats.wilcoxon(nz).pvalue) if nz.size > 0 else 1.0
        extra = {}
    return {"mean": mean, "ci": [float(lo), float(hi)], "p": p, "n": int(d.size), **extra}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        val = (len(ps) - rank) * ps[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


L16, L8, L8NH, SL, FT, BF, RI = "lora_head_r16", "lora_head", "lora", "sequential_linear", "sequential_ft", "bitfit", "lora_head_r16_randinit"
FAMILIES = {
    "seg": [  # Dice, A - B
        ("BraTS: L+h$_{16}$ $-$ SL", "dice", L16, SL, 2),
        ("BraTS: FT $-$ L+h$_{16}$", "dice", FT, L16, 2),
        ("BraTS: L+h$_{16}$ $-$ L+h$_{8}$", "dice", L16, L8, 2),
        ("BraTS: L+h$_{8}$ $-$ L$_{8}$", "dice", L8, L8NH, 2),
        ("BraTS: L+h$_{16}$ $-$ RI", "dice", L16, RI, 2),
        ("ISLES: SL $-$ L+h$_{16}$", "dice", SL + "_t243", L16 + "_t243", 4),
        ("ISLES: FT $-$ L+h$_{16}$", "dice", FT + "_t243", L16 + "_t243", 4),
        ("ISLES: L+h$_{16}$ $-$ RI", "dice", L16 + "_t4", RI.replace("_randinit", "_t4_randinit"), 4),
        ("Hippocampus: SL $-$ L+h$_{16}$", "dice", SL + "_t7", L16 + "_t7", 7),
        ("Hippocampus: FT $-$ L+h$_{16}$", "dice", FT + "_t7", L16 + "_t7", 7),
        ("Hippocampus: L+h$_{16}$ $-$ RI", "dice", L16 + "_t7", RI.replace("_randinit", "_t7_randinit"), 7),
        ("WMH: SL $-$ L+h$_{16}$", "dice", SL + "_t8", L16 + "_t8", 8),
        ("WMH: FT $-$ L+h$_{16}$", "dice", FT + "_t8", L16 + "_t8", 8),
        ("WMH: L+h$_{16}$ $-$ RI", "dice", L16 + "_t8", RI.replace("_randinit", "_t8_randinit"), 8),
    ],
    "age": [  # |error| in years, A - B
        ("IXI: L+h$_{16}$ $-$ constant", "const", L16 + "_t5", "const", 5),
        ("IXI: L+h$_{16}$ $-$ FT", "abserr", L16 + "_t5", FT + "_t5", 5),
        ("IXI: L+h$_{16}$ $-$ RI", "abserr", L16 + "_t5", RI.replace("_randinit", "_t5_randinit"), 5),
        ("OASIS: L+h$_{16}$ $-$ constant", "const", L16 + "_t9", "const", 9),
        ("OASIS: L+h$_{16}$ $-$ FT", "abserr", L16 + "_t9", FT + "_t9", 9),
        ("OASIS: L+h$_{16}$ $-$ RI", "abserr", L16 + "_t9", RI.replace("_randinit", "_t9_randinit"), 9),
    ],
    "cls": [  # accuracy, McNemar
        ("IXI sex: L+h$_{16}$ vs FT", "mcnemar", L16 + "_t6", FT + "_t6", 6),
        ("IXI sex: L+h$_{16}$ vs RI", "mcnemar", L16 + "_t6", RI.replace("_randinit", "_t6_randinit"), 6),
        ("OASIS sex: L+h$_{16}$ vs FT", "mcnemar", L16 + "_t10", FT + "_t10", 10),
        ("OASIS sex: L+h$_{16}$ vs RI", "mcnemar", L16 + "_t10", RI.replace("_randinit", "_t10_randinit"), 10),
        ("OASIS CDR: L+h$_{16}$ vs FT", "mcnemar", L16 + "_t11", FT + "_t11", 11),
    ],
}
# Forgetting (post-sequence) scores are stored only as per-seed aggregates, not per case,
# so BWT contrasts between EWC / LwF / replay and plain fine-tuning are reported as
# mean ± SD over seeds in Table main rather than as paired per-case tests.


def fmt_p(p):
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(np.floor(np.log10(p)))
    m = round(p / 10 ** e)
    if m == 10:          # 9.6e-12 rounds to 10e-12; write 1e-11 instead
        m, e = 1, e + 1
    return f"{m:d}{{\\times}}10^{{{e}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default=str(ROOT / "results" / "stats_v2.json"))
    ap.add_argument("--tab_dir", default=str(ROOT / "IEEE_JBHI_3dBrainLoRA" / "tables"))
    args = ap.parse_args()
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tab_dir).mkdir(parents=True, exist_ok=True)

    out = {}
    for fam, rows in FAMILIES.items():
        res = []
        for label, kind, a, b, task in rows:
            d, nb, nc, used = paired_values(kind, a, b, task)
            s = summarise(kind, d, nb, nc)
            if s is None:
                print(f"  [{fam}] {label}: no data", flush=True)
                continue
            s.update({"label": label, "kind": kind, "A": a, "B": b, "task": task, "seeds": used})
            res.append(s)
        adj = holm([r["p"] for r in res])
        for r, pa in zip(res, adj):
            r["p_holm"] = float(pa)
            print(f"  [{fam}] {r['label']:34s} Δ={r['mean']:+.4f} CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.2e} holm={r['p_holm']:.2e} n={r['n']} seeds={r['seeds']}", flush=True)
        out[fam] = res

        unit = {"seg": "$\\Delta$Dice", "age": "$\\Delta|$err$|$ (yr)", "cls": "$\\Delta$acc", "forget": "$\\Delta$Dice after T2"}[fam]
        lines = []
        for r in res:
            ci = f"$[{r['ci'][0]:+.3f},\\,{r['ci'][1]:+.3f}]$"
            # the single-column global table has no room for an n column; n is in its caption
            ncol = f" & {r['n']}" if fam == "seg" else ""
            lines.append(f"{r['label']} & ${r['mean']:+.3f}$ & {ci} & ${fmt_p(r['p'])}$ & ${fmt_p(r['p_holm'])}${ncol} \\\\")
        (Path(args.tab_dir) / f"paired_{fam}.tex").write_text("\n".join(lines) + "\n")

    json.dump(out, open(args.out_json, "w"), indent=1)
    print(f"wrote {args.out_json} and {args.tab_dir}/paired_*.tex")


if __name__ == "__main__":
    main()
