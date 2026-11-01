#!/usr/bin/env python3
"""
Emit the manuscript's result tables (tabular bodies only) from outputs/benchmark_v2,
mean ± SD over the seeds present, so main.tex can \\input them instead of carrying
hand-copied numbers.

  python scripts/make_tables_v2.py        # -> IEEE_JBHI_3dBrainLoRA/tables/*.tex
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "benchmark_v2"
TAB = ROOT / "IEEE_JBHI_3dBrainLoRA" / "tables"


def runs(n, cfg, seeds=None):
    rs = []
    for f in sorted(glob.glob(str(OUT / n / cfg / "seed*" / "results.json"))):
        if seeds is not None and int(Path(f).parent.name[4:]) not in seeds:
            continue
        rs.append(json.load(open(f)))
    return rs


def vals(n, cfg, task, metric, stage="test", seeds=None):
    v = []
    for r in runs(n, cfg, seeds):
        e = r["per_task"].get(f"task{task}", {})
        if stage in e and metric in e[stage] and e[stage][metric] is not None and np.isfinite(e[stage][metric]):
            v.append(float(e[stage][metric]))
    return np.array(v)


def ms(v, d=3, bold=False):
    if v.size == 0:
        return "n/a"
    s = f"{v.mean():.{d}f}\\pm{v.std(ddof=1) if v.size > 1 else 0:.{d}f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def m(v, d=2):
    return "n/a" if v.size == 0 else f"{v.mean():.{d}f}"


def trainable(n, cfg, task):
    for r in runs(n, cfg):
        return r["per_task"][f"task{task}"]["params"]["trainable"]
    return None


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    nseeds = {cfg: len(runs("n32", cfg)) for cfg in ("lora_head_r16", "sequential_ft_t32", "lora_head_r16_t9")}
    print("seeds:", nseeds)

    # ---- tab:main -------------------------------------------------------------
    rows = [("BitFit", "bitfit"), ("LoRA (enc+dec), $r{=}8$", "lora"), ("LoRA + head, $r{=}4$", "lora_head_r4"),
            ("LoRA + head, $r{=}8$", "lora_head"), ("LoRA + head, $r{=}16$", "lora_head_r16"), ("Sequential Linear", "sequential_linear")]
    seq = [("Sequential FT", "sequential_ft"), ("EWC", "ewc"), ("LwF", "lwf"), ("Replay", "replay")]
    L = []
    L.append("\\multicolumn{8}{l}{\\emph{Frozen backbone (task-isolated; BWT $=0$ by construction)}} \\\\")
    for lab, c in rows:
        dice, hd, mae, r2 = vals("n32", c, 2, "dice"), vals("n32", c, 2, "hd95"), vals("n32", c, 3, "mae"), vals("n32", c, 3, "r2")
        after = "= T1" if c != "sequential_linear" else ms(vals("n32", c, 2, "dice", "test_after_task3"))
        L.append(f"{lab:26s} & {ms(dice)} & {ms(hd, 1)} & {ms(mae, 2)} & ${r2.mean():+.2f}$ & {after} & 0.000 & {trainable('n32', c, 2):,} \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{8}{l}{\\emph{Backbone updated per task}} \\\\")
    for lab, c in seq:
        dice, hd, mae, r2 = vals("n32", c, 2, "dice"), vals("n32", c, 2, "hd95"), vals("n32", c, 3, "mae"), vals("n32", c, 3, "r2")
        aft = vals("n32", c, 2, "dice", "test_after_task3"); bwt = aft - dice[: aft.size]
        L.append(f"{lab:26s} & {ms(dice)} & {ms(hd, 1)} & {ms(mae, 2)} & ${r2.mean():+.2f}$ & {ms(aft)} & {ms(bwt)} & {trainable('n32', c, 2):,} \\\\")
    (TAB / "main_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:rank ----------------------------------------------------------------
    L = []
    for lab, c in (("BitFit", "bitfit"), ("$r=4$", "lora_head_r4"), ("$r=8$", "lora_head"), ("$r=16$", "lora_head_r16")):
        L.append(f"{lab:8s} & {ms(vals('n32', c, 2, 'dice'), bold=(c == 'lora_head_r16'))} & {ms(vals('n32', c, 2, 'hd95'), 1, bold=(c == 'lora_head_r16'))} & {m(vals('n32', c, 3, 'mae'))} & {trainable('n32', c, 2):,} \\\\")
    L.append("\\midrule")
    c = "sequential_linear"
    L.append(f"Seq.\\ Linear & {ms(vals('n32', c, 2, 'dice'))} & {ms(vals('n32', c, 2, 'hd95'), 1)} & {m(vals('n32', c, 3, 'mae'))} & {trainable('n32', c, 2):,} \\\\")
    (TAB / "rank_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:shot ----------------------------------------------------------------
    L = []
    for lab, c in (("LoRA + head, $r{=}16$", "lora_head_r16"), ("Sequential Linear", "sequential_linear")):
        L.append(f"{lab:22s} & " + " & ".join(ms(vals(n, c, 2, "dice")) for n in ("n16", "n32", "n64")) + " \\\\")
    (TAB / "shot_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:age (IXI whole head vs patch) ---------------------------------------
    L = []
    for lab, c5, c3 in (("LoRA + head, $r{=}16$", "lora_head_r16_t5", "lora_head_r16"), ("Sequential FT", "sequential_ft_t5", "sequential_ft"),
                        ("Sequential Linear", "sequential_linear_t5", "sequential_linear"), ("BitFit", "bitfit_t5", "bitfit")):
        b = c5 == "lora_head_r16_t5"
        L.append(f"{lab:22s} & {ms(vals('n32', c5, 5, 'mae'), 2, b)} & {ms(vals('n32', c5, 5, 'r2'), 2, b).replace('$', '$+', 1) if vals('n32', c5, 5, 'r2').mean() >= 0 else ms(vals('n32', c5, 5, 'r2'), 2)} & {m(vals('n32', c5, 5, 'pred_std'))} & {m(vals('n32', c3, 3, 'mae'))} \\\\")
    (TAB / "age_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:all (every single task, four strategies + random-init) --------------
    def cell(c, task, metric, d, best=False, const_key=None):
        v = vals("n32", c, task, metric)
        if v.size == 0:
            return "n/a"
        if const_key is not None:
            cv = vals("n32", c, task, const_key)
            if cv.size and ((const_key == "pred_std" and cv.mean() < 0.5) or (const_key == "balanced_acc" and abs(cv.mean() - 0.5) < 0.02)):
                return f"{v.mean():.{d}f} (const.)"
        return ms(v, d, best)
    groups = [
        ("\\emph{Tasks with a large task-specific decoder (Dice$\\uparrow$)}", [
            ("T1 BraTS glioma", "seg, 3\\,ch", 2, "dice", 3, {"L": "lora_head_r16", "SL": "sequential_linear", "FT": "sequential_ft", "BF": "bitfit", "AD": "adapter", "RI": "lora_head_r16_randinit"}),
            ("T3 ISLES stroke", "seg, 3\\,ch", 4, "dice", 3, {"L": "lora_head_r16_t243", "SL": "sequential_linear_t243", "FT": "sequential_ft_t243", "BF": "bitfit_t4", "AD": "adapter_t4", "RI": "lora_head_r16_t4_randinit"}),
            ("T4 MSD hippocampus", "seg, 1\\,ch", 7, "dice", 3, {"L": "lora_head_r16_t7", "SL": "sequential_linear_t7", "FT": "sequential_ft_t7", "BF": "bitfit_t7", "AD": "adapter_t7", "RI": "lora_head_r16_t7_randinit"}),
            ("T6 WMH 2017", "seg, 2\\,ch, 5 scanners", 8, "dice", 3, {"L": "lora_head_r16_t8", "SL": "sequential_linear_t8", "FT": "sequential_ft_t8", "BF": "bitfit_t8", "AD": "adapter_t8", "RI": "lora_head_r16_t8_randinit"}),
        ]),
        ("\\emph{Tasks with only a $\\sim$1K-parameter head}", [
            ("T2 IXI brain age", "reg, MAE yr$\\downarrow$", 5, "mae", 2, {"L": "lora_head_r16_t5", "SL": "sequential_linear_t5", "FT": "sequential_ft_t5", "BF": "bitfit_t5", "AD": "adapter_t5", "RI": "lora_head_r16_t5_randinit"}),
            ("T7 OASIS brain age", "reg, MAE yr$\\downarrow$", 9, "mae", 2, {"L": "lora_head_r16_t9", "SL": "sequential_linear_t9", "FT": "sequential_ft_t9", "BF": "bitfit_t9", "AD": "adapter_t9", "RI": "lora_head_r16_t9_randinit"}),
            ("T5 IXI sex", "cls, AUC$\\uparrow$", 6, "auc", 3, {"L": "lora_head_r16_t6", "SL": "sequential_linear_t6", "FT": "sequential_ft_t6", "BF": "bitfit_t6", "AD": "adapter_t6", "RI": "lora_head_r16_t6_randinit"}),
            ("T8 OASIS sex", "cls, AUC$\\uparrow$", 10, "auc", 3, {"L": "lora_head_r16_t10", "SL": "sequential_linear_t10", "FT": "sequential_ft_t10", "BF": "bitfit_t10", "AD": "adapter_t10", "RI": "lora_head_r16_t10_randinit"}),
            ("T9 OASIS CDR$>$0", "cls, AUC$\\uparrow$", 11, "auc", 3, {"L": "lora_head_r16_t11", "SL": "sequential_linear_t11", "FT": "sequential_ft_t11", "BF": "bitfit_t11", "AD": "adapter_t11", "RI": "lora_head_r16_t11_randinit"}),
        ]),
    ]
    L = []
    for head, items in groups:
        L.append(f"\\multicolumn{{8}}{{l}}{{{head}}} \\\\")
        for name, typ, task, metric, d, cf in items:
            const_key = "pred_std" if metric == "mae" else ("balanced_acc" if metric == "auc" else None)
            means = {k: vals("n32", c, task, metric).mean() if vals("n32", c, task, metric).size else np.nan for k, c in cf.items()}
            cand = {k: v for k, v in means.items() if k in ("L", "SL", "FT", "BF", "AD") and np.isfinite(v)}
            best = (min if metric == "mae" else max)(cand, key=cand.get) if cand else None
            cells = [cell(cf[k], task, metric, d, best=(k == best), const_key=const_key) for k in ("L", "SL", "FT", "BF", "AD", "RI")]
            L.append(f"{name:22s} & {typ} & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule")
    L.pop()
    (TAB / "all_tasks_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:three (forward, 5 seeds) + reversed (3 seeds) -----------------------
    L = []
    for lab, c in (("LoRA + head, $r{=}16$", "lora_head_r16_t243"), ("Sequential Linear", "sequential_linear_t243"), ("Sequential FT", "sequential_ft_t243")):
        b2, b4, a3 = vals("n32", c, 2, "dice"), vals("n32", c, 4, "dice"), vals("n32", c, 3, "mae")
        if c.startswith("lora"):
            tail = "= & = & ="
        else:
            tail = f"{ms(vals('n32', c, 2, 'dice', 'test_after_task4'))} & {ms(vals('n32', c, 2, 'dice', 'test_after_task3'))} & {ms(vals('n32', c, 4, 'dice', 'test_after_task3'))}"
        L.append(f"{lab:22s} & {ms(b2)} & {ms(b4)} & {ms(a3, 2)} & {tail} \\\\")
    (TAB / "three_body.tex").write_text("\n".join(L) + "\n")
    L = []
    for lab, c in (("LoRA + head, $r{=}16$", "lora_head_r16_t342"), ("Sequential FT", "sequential_ft_t342")):
        a3, b4, b2 = vals("n32", c, 3, "mae"), vals("n32", c, 4, "dice"), vals("n32", c, 2, "dice")
        if c.startswith("lora"):
            tail = "= & = & ="
        else:
            tail = f"{ms(vals('n32', c, 3, 'mae', 'test_after_task4'), 2)} & {ms(vals('n32', c, 3, 'mae', 'test_after_task2'), 2)} & {ms(vals('n32', c, 4, 'dice', 'test_after_task2'))}"
        L.append(f"{lab:22s} & {ms(a3, 2)} & {ms(b4)} & {ms(b2)} & {tail} \\\\")
    (TAB / "three_rev_body.tex").write_text("\n".join(L) + "\n")

    # ---- two-task reversed (age -> BraTS), 3 seeds ------------------------------
    L = []
    for lab, c in (("Sequential Linear", "sequential_linear_t32"), ("Sequential FT", "sequential_ft_t32"), ("EWC", "ewc_t32"), ("LwF", "lwf_t32"), ("Replay", "replay_t32")):
        a, aa, b = vals("n32", c, 3, "mae"), vals("n32", c, 3, "mae", "test_after_task2"), vals("n32", c, 2, "dice")
        L.append(f"{lab:18s} & {ms(a, 2)} & {ms(aa, 2)} & {ms(b)} \\\\")
    (TAB / "two_rev_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:full (full-data reference, seed 42) ---------------------------------
    def fcell(c, task, metric, d):
        v = vals("nfull", c, task, metric)
        return "n/a" if v.size == 0 else f"{v.mean():.{d}f}"
    L = []
    for name, typ, task, metric, d, ntr, c32, cfull in (
            ("T1 BraTS glioma", "Dice$\\uparrow$", 2, "dice", 3, 908, {"L": "lora_head_r16", "SL": "sequential_linear", "FT": "sequential_ft"}, {"L": "lora_head_r16_t2", "SL": "sequential_linear_t2", "FT": "sequential_ft_t2"}),
            ("T2 IXI brain age", "MAE yr$\\downarrow$", 5, "mae", 2, 249, {"L": "lora_head_r16_t5", "SL": "sequential_linear_t5", "FT": "sequential_ft_t5"}, {"L": "lora_head_r16_t5", "SL": None, "FT": "sequential_ft_t5"}),
            ("T5 IXI sex", "AUC$\\uparrow$", 6, "auc", 3, 249, {"L": "lora_head_r16_t6", "SL": "sequential_linear_t6", "FT": "sequential_ft_t6"}, {"L": "lora_head_r16_t6", "SL": None, "FT": "sequential_ft_t6"}),
            ("T7 OASIS brain age", "MAE yr$\\downarrow$", 9, "mae", 2, 106, {"L": "lora_head_r16_t9", "SL": "sequential_linear_t9", "FT": "sequential_ft_t9"}, {"L": "lora_head_r16_t9", "SL": None, "FT": "sequential_ft_t9"}),
            ("T8 OASIS sex", "AUC$\\uparrow$", 10, "auc", 3, 106, {"L": "lora_head_r16_t10", "SL": "sequential_linear_t10", "FT": "sequential_ft_t10"}, {"L": "lora_head_r16_t10", "SL": None, "FT": "sequential_ft_t10"})):
        cells = []
        for k in ("L", "SL", "FT"):
            cells.append(ms(vals("n32", c32[k], task, metric), d))
            cells.append(fcell(cfull[k], task, metric, d) if cfull[k] else "n/a")
        L.append(f"{name:20s} & {typ} & {ntr} & " + " & ".join(cells) + " \\\\")
    (TAB / "full_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:swin (second backbone: Swin UNETR, seeds 42-44 on both backbones) ------
    SW = (42, 43, 44)
    def scell(c, task, metric, d, const_key, best):
        v = vals("n32", c, task, metric, seeds=SW)
        if v.size == 0:
            return "n/a"
        if const_key is not None:
            cv = vals("n32", c, task, const_key, seeds=SW)
            if cv.size and ((const_key == "pred_std" and cv.mean() < 0.5) or (const_key == "balanced_acc" and abs(cv.mean() - 0.5) < 0.02)):
                return f"{v.mean():.{d}f} (const.)"
        return ms(v, d, best)
    cols = (  # (task, metric, decimals, const marker, higher-is-better, unet suffix, swin suffix)
        (2, "dice", 3, None, True, "", "_t2"),
        (5, "mae", 2, "pred_std", False, "_t5", "_t5"),
        (6, "auc", 3, "balanced_acc", True, "_t6", "_t6"))
    strategies = (("LoRA + head ($r{=}16$)", "lora_head_r16"), ("Sequential Linear", "sequential_linear"),
                  ("Sequential FT", "sequential_ft"), ("BitFit", "bitfit"))
    best = {}
    for task, metric, d, ck, hi, su, ss in cols:
        for bb, suf in (("", su), ("swin_", ss)):
            means = {c: vals("n32", f"{bb}{c}{suf}", task, metric, seeds=SW).mean()
                     for _, c in strategies if vals("n32", f"{bb}{c}{suf}", task, metric, seeds=SW).size}
            if means:
                best[(task, bb)] = (max if hi else min)(means, key=means.get)
    L = []
    for lab, c in strategies:
        cells = []
        for task, metric, d, ck, hi, su, ss in cols:
            for bb, suf in (("", su), ("swin_", ss)):
                cells.append(scell(f"{bb}{c}{suf}", task, metric, d, ck, best.get((task, bb)) == c))
        L.append(f"{lab:24s} & " + " & ".join(cells) + " \\\\")
    tr = []
    for task, metric, d, ck, hi, su, ss in cols:
        for bb, suf in (("", su), ("swin_", ss)):
            cells_t = [trainable("n32", f"{bb}{c}{suf}", task) for _, c in strategies]
            tr.append(cells_t)
    (TAB / "swin_body.tex").write_text("\n".join(L) + "\n")
    # trainable-parameter line: LoRA / probe / FT / BitFit per backbone, segmentation and whole-head
    def k(x):
        return "n/a" if x is None else (f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.1f}K")
    seg_u = [trainable("n32", c, 2) for _, c in strategies]
    seg_s = [trainable("n32", f"swin_{c}_t2", 2) for _, c in strategies]
    glb_u = [trainable("n32", f"{c}_t5", 5) for _, c in strategies]
    glb_s = [trainable("n32", f"swin_{c}_t5", 5) for _, c in strategies]
    (TAB / "swin_params.tex").write_text(
        "Trainable, segmentation & " + " & ".join(f"{k(a)} / {k(b)}" for a, b in zip(seg_u, seg_s)) + " \\\\\n"
        "Trainable, whole head & " + " & ".join(f"{k(a)} / {k(b)}" for a, b in zip(glb_u, glb_s)) + " \\\\\n")

    n_runs = len(glob.glob(str(OUT / "n*" / "*" / "seed*" / "results.json")))
    (TAB / "counts.tex").write_text(f"\\newcommand{{\\nruns}}{{{n_runs}}}\n")
    print("runs:", n_runs, "| tables written to", TAB)


if __name__ == "__main__":
    main()
