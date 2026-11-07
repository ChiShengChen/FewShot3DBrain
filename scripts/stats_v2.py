#!/usr/bin/env python3
"""
Paired statistics for the JBHI manuscript, with the training run (seed) as the unit of inference.

Every contrast pairs two configurations inside a seed (all methods share the split within a
seed). The per-seed statistic is the mean paired per-case difference for that seed (or the
difference of per-seed AUCs / BWTs); the five (or three) per-seed values are then summarised
by their mean, a t-based 95% CI, a two-sided paired t-test against zero, the number of seeds
favouring the first method, and a Holm adjustment within the family (the table the contrast
feeds). The earlier per-case pooling (Wilcoxon on ~1,250 case differences from five runs,
with the same subjects recurring across seeds) treated pseudo-replicates as independent and is
kept only as a secondary descriptive statistic in the JSON output.

  python scripts/stats_v2.py                    # writes results/stats_v2.json + tables/paired_*.tex

Kinds:  dice     per-case Dice, A - B, averaged within seed
        abserr   per-case |pred - gt| in years, A - B, averaged within seed (negative favours A)
        const    per-case |pred - gt| of A minus |mean(train ages) - gt| (A vs training-mean constant)
        auc      per-seed AUC(A) - AUC(B) from the per-case scores (positive favours A)
        bwt      per-seed backward transfer on BraTS after the two-task sequence, A - B
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


def per_case(n, cfg, seed, task):
    f = OUT / n / cfg / f"seed{seed}" / f"task{task}_per_case.json"
    if not f.exists():
        return None
    return {os.path.basename(c["case"]): c for c in json.load(open(f))}


def results_json(n, cfg, seed):
    f = OUT / n / cfg / f"seed{seed}" / "results.json"
    return json.load(open(f)) if f.exists() else None


def train_mean_label(task, seed, n_shot=32):
    sp = json.load(open(ROOT / "data/splits_v3" / f"{TASK_CONFIGS[task]['name']}_n{n_shot}_seed{seed}.json"))["splits"]
    vals = [float(open(p + ".txt").read().split()[0]) for p in sp["train"]]
    return float(np.mean(vals))


def auc_from_scores(scores, gts):
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    s, g = np.asarray(scores, float), np.asarray(gts, int)
    pos, neg = s[g == 1], s[g == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def seed_values(kind, cfgA, cfgB, task, n="n32", seeds=SEEDS):
    """Return per-seed statistics [(seed, d_s, n_cases)] and the pooled per-case differences."""
    per_seed, pooled = [], []
    for s in seeds:
        if kind == "bwt":
            rA, rB = results_json(n, cfgA, s), results_json(n, cfgB, s)
            if rA is None or rB is None:
                continue
            def bwt(r):
                e = r["per_task"][f"task{task}"]
                after = [v for k, v in e.items() if k.startswith("test_after_task")]
                return float(after[-1]["dice"]) - float(e["test"]["dice"])
            per_seed.append((s, bwt(rA) - bwt(rB), 1))
            continue
        A = per_case(n, cfgA, s, task)
        B = per_case(n, cfgB, s, task) if cfgB != "const" else None
        if A is None or (cfgB != "const" and B is None):
            continue
        if kind == "const":
            m = train_mean_label(task, s)
            d = [abs(a["pred"] - a["gt"]) - abs(m - a["gt"]) for a in A.values()]
        else:
            keys = sorted(A.keys() & B.keys())
            if kind == "dice":
                d = [A[k]["dice"] - B[k]["dice"] for k in keys]
            elif kind == "abserr":
                d = [abs(A[k]["pred"] - A[k]["gt"]) - abs(B[k]["pred"] - B[k]["gt"]) for k in keys]
            elif kind == "auc":
                d = [auc_from_scores([A[k]["score"] for k in keys], [A[k]["gt"] for k in keys])
                     - auc_from_scores([B[k]["score"] for k in keys], [B[k]["gt"] for k in keys])]
            else:
                raise ValueError(kind)
        d = np.asarray(d, float)
        per_seed.append((s, float(d.mean()), int(d.size)))
        if kind != "auc":
            pooled.extend(d.tolist())
    return per_seed, np.asarray(pooled, float)


def summarise(per_seed, pooled):
    if not per_seed:
        return None
    ds = np.array([d for _, d, _ in per_seed])
    n = ds.size
    mean, sd = float(ds.mean()), float(ds.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        se = sd / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        ci = [mean - tcrit * se, mean + tcrit * se]
        p = float(stats.ttest_1samp(ds, 0.0).pvalue)
    else:  # identical per-seed values (e.g. a method that is constant on every seed)
        ci = [mean, mean]
        p = 1.0 if abs(mean) < 1e-12 else 0.0
    out = {"mean": mean, "sd_seeds": sd, "ci": [float(ci[0]), float(ci[1])], "p": p,
           "n_seeds": int(n), "seeds_favouring_A": int((ds > 0).sum()), "seeds_against_A": int((ds < 0).sum()),
           "per_seed": {str(s): d for s, d, _ in per_seed}, "n_cases": int(sum(c for _, _, c in per_seed))}
    if pooled.size:
        nz = pooled[pooled != 0]
        out["percase"] = {"mean": float(pooled.mean()), "n": int(pooled.size),
                          "wilcoxon_p": float(stats.wilcoxon(nz).pvalue) if nz.size else 1.0}
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


L16, L8, L8NH, SL, FT, BF, RI, AD = "lora_head_r16", "lora_head", "lora", "sequential_linear", "sequential_ft", "bitfit", "lora_head_r16_randinit", "adapter"
FAMILIES = {
    "seg": [  # Dice, A - B
        ("BraTS: L+h$_{16}$ $-$ SL", "dice", L16, SL, 2),
        ("BraTS: FT $-$ L+h$_{16}$", "dice", FT, L16, 2),
        ("BraTS: L+h$_{16}$ $-$ L+h$_{8}$", "dice", L16, L8, 2),
        ("BraTS: L+h$_{8}$ $-$ L$_{8}$", "dice", L8, L8NH, 2),
        ("BraTS: L+h$_{16}$ $-$ RI", "dice", L16, RI, 2),
        ("BraTS: L+h$_{16}$ $-$ AD", "dice", L16, AD, 2),
        ("ISLES: SL $-$ L+h$_{16}$", "dice", SL + "_t4", L16 + "_t4", 4),
        ("ISLES: FT $-$ L+h$_{16}$", "dice", FT + "_t4", L16 + "_t4", 4),
        ("ISLES: L+h$_{16}$ $-$ RI", "dice", L16 + "_t4", RI.replace("_randinit", "_t4_randinit"), 4),
        ("ISLES: L+h$_{16}$ $-$ AD", "dice", L16 + "_t4", AD + "_t4", 4),
        ("ISLES: L+h$_{16}$ single $-$ seq.", "dice", L16 + "_t4", L16 + "_t243", 4),
        ("Hippocampus: SL $-$ L+h$_{16}$", "dice", SL + "_t7", L16 + "_t7", 7),
        ("Hippocampus: FT $-$ L+h$_{16}$", "dice", FT + "_t7", L16 + "_t7", 7),
        ("Hippocampus: L+h$_{16}$ $-$ RI", "dice", L16 + "_t7", RI.replace("_randinit", "_t7_randinit"), 7),
        ("Hippocampus: L+h$_{16}$ $-$ AD", "dice", L16 + "_t7", AD + "_t7", 7),
        ("WMH: SL $-$ L+h$_{16}$", "dice", SL + "_t8", L16 + "_t8", 8),
        ("WMH: FT $-$ L+h$_{16}$", "dice", FT + "_t8", L16 + "_t8", 8),
        ("WMH: L+h$_{16}$ $-$ RI", "dice", L16 + "_t8", RI.replace("_randinit", "_t8_randinit"), 8),
        ("WMH: L+h$_{16}$ $-$ AD", "dice", L16 + "_t8", AD + "_t8", 8),
    ],
    "age": [  # |error| in years, A - B
        ("IXI: L+h$_{16}$ $-$ constant", "const", L16 + "_t5", "const", 5),
        ("IXI: L+h$_{16}$ $-$ FT", "abserr", L16 + "_t5", FT + "_t5", 5),
        ("IXI: L+h$_{16}$ $-$ RI", "abserr", L16 + "_t5", RI.replace("_randinit", "_t5_randinit"), 5),
        ("IXI: L+h$_{16}$ $-$ AD", "abserr", L16 + "_t5", AD + "_t5", 5),
        ("OASIS: L+h$_{16}$ $-$ constant", "const", L16 + "_t9", "const", 9),
        ("OASIS: L+h$_{16}$ $-$ FT", "abserr", L16 + "_t9", FT + "_t9", 9),
        ("OASIS: L+h$_{16}$ $-$ RI", "abserr", L16 + "_t9", RI.replace("_randinit", "_t9_randinit"), 9),
        ("OASIS: L+h$_{16}$ $-$ AD", "abserr", L16 + "_t9", AD + "_t9", 9),
    ],
    "cls": [  # AUC, A - B
        ("IXI sex: L+h$_{16}$ $-$ FT", "auc", L16 + "_t6", FT + "_t6", 6),
        ("IXI sex: L+h$_{16}$ $-$ SL", "auc", L16 + "_t6", SL + "_t6", 6),
        ("IXI sex: L+h$_{16}$ $-$ RI", "auc", L16 + "_t6", RI.replace("_randinit", "_t6_randinit"), 6),
        ("IXI sex: L+h$_{16}$ $-$ AD", "auc", L16 + "_t6", AD + "_t6", 6),
        ("OASIS sex: L+h$_{16}$ $-$ FT", "auc", L16 + "_t10", FT + "_t10", 10),
        ("OASIS sex: L+h$_{16}$ $-$ SL", "auc", L16 + "_t10", SL + "_t10", 10),
        ("OASIS sex: L+h$_{16}$ $-$ RI", "auc", L16 + "_t10", RI.replace("_randinit", "_t10_randinit"), 10),
        ("OASIS sex: L+h$_{16}$ $-$ AD", "auc", L16 + "_t10", AD + "_t10", 10),
        ("OASIS CDR: L+h$_{16}$ $-$ FT", "auc", L16 + "_t11", FT + "_t11", 11),
    ],
    "forget": [  # BWT on BraTS after BraTS -> patch brain age (T2 not learned), A - B
        ("BraTS BWT: EWC $-$ FT", "bwt", "ewc", FT, 2),
        ("BraTS BWT: LwF $-$ FT", "bwt", "lwf", FT, 2),
        ("BraTS BWT: replay $-$ FT", "bwt", "replay", FT, 2),
    ],
    "forget25": [  # BWT on BraTS after BraTS -> whole-head brain age; same first-task model
        ("EWC ($\\lambda{=}10^2$) $-$ FT", "bwt", "ewc_t25_ewc100", FT + "_t25", 2),
        ("EWC ($\\lambda{=}10^3$) $-$ FT", "bwt", "ewc_t25", FT + "_t25", 2),
        ("EWC ($\\lambda{=}10^4$) $-$ FT", "bwt", "ewc_t25_ewc10000", FT + "_t25", 2),
        ("LwF, features $-$ FT", "bwt", "lwf_t25", FT + "_t25", 2),
        ("LwF, outputs $-$ FT", "bwt", "lwf_out_t25", FT + "_t25", 2),
        ("Replay, features $-$ FT", "bwt", "replay_t25", FT + "_t25", 2),
        ("Replay, labels $-$ FT", "bwt", "replay_label_t25", FT + "_t25", 2),
    ],
    "controls": [  # breaking the head-size / task-type confound; A - LoRA + head
        ("BraTS: head only $-$ L+h$_{16}$", "dice", "head_only_t2", L16, 2),
        ("ISLES: head only $-$ L+h$_{16}$", "dice", "head_only_t4", L16 + "_t4", 4),
        ("Hippocampus: head only $-$ L+h$_{16}$", "dice", "head_only_t7", L16 + "_t7", 7),
        ("WMH: head only $-$ L+h$_{16}$", "dice", "head_only_t8", L16 + "_t8", 8),
        ("BraTS: partial FT $-$ L+h$_{16}$", "dice", "partial_ft_t2", L16, 2),
        ("IXI age: ridge probe $-$ constant", "const", "ridge_probe_t5", "const", 5),
        ("IXI age: ridge probe $-$ L+h$_{16}$", "abserr", "ridge_probe_t5", L16 + "_t5", 5),
        ("IXI age: MLP head $-$ L+h$_{16}$", "abserr", "mlp_head_t5", L16 + "_t5", 5),
        ("IXI age: partial FT $-$ L+h$_{16}$", "abserr", "partial_ft_t5", L16 + "_t5", 5),
        ("OASIS age: ridge probe $-$ constant", "const", "ridge_probe_t9", "const", 9),
        ("OASIS age: ridge probe $-$ L+h$_{16}$", "abserr", "ridge_probe_t9", L16 + "_t9", 9),
        ("OASIS age: MLP head $-$ L+h$_{16}$", "abserr", "mlp_head_t9", L16 + "_t9", 9),
        ("OASIS age: partial FT $-$ L+h$_{16}$", "abserr", "partial_ft_t9", L16 + "_t9", 9),
        ("IXI sex: ridge probe $-$ L+h$_{16}$", "auc", "ridge_probe_t6", L16 + "_t6", 6),
        ("IXI sex: MLP head $-$ L+h$_{16}$", "auc", "mlp_head_t6", L16 + "_t6", 6),
        ("IXI sex: partial FT $-$ L+h$_{16}$", "auc", "partial_ft_t6", L16 + "_t6", 6),
        ("OASIS sex: ridge probe $-$ L+h$_{16}$", "auc", "ridge_probe_t10", L16 + "_t10", 10),
        ("OASIS sex: MLP head $-$ L+h$_{16}$", "auc", "mlp_head_t10", L16 + "_t10", 10),
        ("OASIS sex: partial FT $-$ L+h$_{16}$", "auc", "partial_ft_t10", L16 + "_t10", 10),
    ],
    "pretrain": [  # pretrained checkpoint minus random initialisation, same configuration
        ("BraTS, probe: pretrained $-$ random", "dice", SL, SL + "_t2_randinit", 2),
        ("BraTS, full FT: pretrained $-$ random", "dice", FT, FT + "_t2_randinit", 2),
        ("IXI age, full FT: pretrained $-$ random", "abserr", FT + "_t5", FT + "_t5_randinit", 5),
        ("IXI age, ridge: pretrained $-$ random", "abserr", "ridge_probe_t5", "ridge_probe_t5_randinit", 5),
        ("IXI sex, ridge: pretrained $-$ random", "auc", "ridge_probe_t6", "ridge_probe_t6_randinit", 6),
        ("OASIS age, ridge: pretrained $-$ random", "abserr", "ridge_probe_t9", "ridge_probe_t9_randinit", 9),
        ("OASIS sex, ridge: pretrained $-$ random", "auc", "ridge_probe_t10", "ridge_probe_t10_randinit", 10),
    ],
    "swin": [  # second backbone (Swin UNETR), seeds 42-44 only
        ("BraTS (Swin): L+h$_{16}$ $-$ SL", "dice", "swin_" + L16 + "_t2", "swin_" + SL + "_t2", 2),
        ("BraTS (Swin): FT $-$ L+h$_{16}$", "dice", "swin_" + FT + "_t2", "swin_" + L16 + "_t2", 2),
        ("BraTS (Swin): L+h$_{16}$ $-$ BF", "dice", "swin_" + L16 + "_t2", "swin_" + BF + "_t2", 2),
        ("IXI age (Swin): L+h$_{16}$ $-$ constant", "const", "swin_" + L16 + "_t5", "const", 5),
        ("IXI sex (Swin): L+h$_{16}$ $-$ SL", "auc", "swin_" + L16 + "_t6", "swin_" + SL + "_t6", 6),
        ("IXI sex (Swin): L+h$_{16}$ $-$ FT", "auc", "swin_" + L16 + "_t6", "swin_" + FT + "_t6", 6),
        ("IXI sex (Swin): L+h$_{16}$ $-$ BF", "auc", "swin_" + L16 + "_t6", "swin_" + BF + "_t6", 6),
    ],
}
DECIMALS = {"dice": 3, "auc": 3, "bwt": 3, "abserr": 2, "const": 2}


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(np.floor(np.log10(p)))
    m = round(p / 10 ** e)
    if m == 10:
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
            per_seed, pooled = seed_values(kind, a, b, task, seeds=(42, 43, 44) if fam == "swin" else SEEDS)
            s = summarise(per_seed, pooled)
            if s is None:
                print(f"  [{fam}] {label}: no data", flush=True)
                continue
            s.update({"label": label, "kind": kind, "A": a, "B": b, "task": task})
            res.append(s)
        adj = holm([r["p"] for r in res])
        for r, pa in zip(res, adj):
            r["p_holm"] = float(pa)
            pc = r.get("percase", {})
            print(f"  [{fam}] {r['label']:40s} Δ={r['mean']:+.4f} CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
                  f"p={r['p']:.3f} holm={r['p_holm']:.3f} sign={r['seeds_favouring_A']}/{r['n_seeds']}"
                  + (f"  (per-case pooled p={pc['wilcoxon_p']:.1e}, n={pc['n']})" if pc else ""), flush=True)
        out[fam] = res

        lines = []
        for r in res:
            d = DECIMALS[r["kind"]]
            ci = f"$[{r['ci'][0]:+.{d}f},\\,{r['ci'][1]:+.{d}f}]$"
            vals = list(r["per_seed"].values())
            rng = f"${min(vals):+.{d}f}$ to ${max(vals):+.{d}f}$"
            sign = f"{r['seeds_favouring_A']}/{r['n_seeds']}"
            lines.append(f"{r['label']} & ${r['mean']:+.{d}f}$ & {ci} & {rng} & ${fmt_p(r['p'])}$ & ${fmt_p(r['p_holm'])}$ & {sign} \\\\")
        (Path(args.tab_dir) / f"paired_{fam}.tex").write_text("\n".join(lines) + "\n")

    json.dump(out, open(args.out_json, "w"), indent=1)
    print(f"wrote {args.out_json} and {args.tab_dir}/paired_*.tex")


if __name__ == "__main__":
    main()
