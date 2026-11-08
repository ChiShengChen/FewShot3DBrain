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
TAB = Path(__import__('os').environ.get('TAB_DIR', str(ROOT / 'IEEE_JBHI_3dBrainLoRA' / 'tables')))


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


def const_count(n, cfg, task, const_key, seeds=None):
    """(k, n): how many seeds return a constant output (pred SD < 0.5 yr, or balanced
    accuracy within 0.02 of 0.5), out of the seeds present."""
    cv = vals(n, cfg, task, const_key, seeds=seeds)
    if cv.size == 0:
        return 0, 0
    k = int((cv < 0.5).sum()) if const_key == "pred_std" else int((np.abs(cv - 0.5) < 0.02).sum())
    return k, int(cv.size)


def const_cell(v, d, k, n, best=False):
    """Mean +/- SD, or the mean marked const. when every seed is constant, or the count of
    constant seeds when only some are."""
    if n and k == n:
        return f"{v.mean():.{d}f}$^{{\\mathrm{{c}}}}$"
    s = ms(v, d, best)
    if not k:
        return s
    # superscript c k/n inside the closing math of ms(): "$...$" -> "$...^{c k/n}$"
    assert s.endswith("$")
    return s[:-1] + f"^{{\\mathrm{{c}}\\,{k}/{n}}}$"


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
            k, n = const_count("n32", c, task, const_key)
            return const_cell(v, d, k, n, best)
        return ms(v, d, best)
    groups = [
        ("\\emph{Tasks with a large task-specific decoder (Dice$\\uparrow$)}", [
            ("T1 BraTS glioma", "seg, 3\\,ch", 2, "dice", 3, {"L": "lora_head_r16", "SL": "sequential_linear", "FT": "sequential_ft", "BF": "bitfit", "AD": "adapter", "RI": "lora_head_r16_randinit"}),
            ("T3 ISLES stroke", "seg, 3\\,ch", 4, "dice", 3, {"L": "lora_head_r16_t4", "SL": "sequential_linear_t4", "FT": "sequential_ft_t4", "BF": "bitfit_t4", "AD": "adapter_t4", "RI": "lora_head_r16_t4_randinit"}),
            ("T4 MSD hippocampus", "seg, 1\\,ch", 7, "dice", 3, {"L": "lora_head_r16_t7", "SL": "sequential_linear_t7", "FT": "sequential_ft_t7", "BF": "bitfit_t7", "AD": "adapter_t7", "RI": "lora_head_r16_t7_randinit"}),
            ("T6 WMH 2017", "seg, 2\\,ch", 8, "dice", 3, {"L": "lora_head_r16_t8", "SL": "sequential_linear_t8", "FT": "sequential_ft_t8", "BF": "bitfit_t8", "AD": "adapter_t8", "RI": "lora_head_r16_t8_randinit"}),
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
            # A method that is constant on half or more of its seeds is not eligible for bold.
            cand = {k: v for k, v in means.items() if k in ("L", "SL", "FT", "BF", "AD") and np.isfinite(v)
                    and (const_key is None or const_count("n32", cf[k], task, const_key)[0] * 2 < max(1, const_count("n32", cf[k], task, const_key)[1]))}
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
            ("T2 IXI brain age", "MAE yr$\\downarrow$", 5, "mae", 2, 249, {"L": "lora_head_r16_t5", "SL": "sequential_linear_t5", "FT": "sequential_ft_t5"}, {"L": "lora_head_r16_t5", "SL": "sequential_linear_t5", "FT": "sequential_ft_t5"}),
            ("T5 IXI sex", "AUC$\\uparrow$", 6, "auc", 3, 249, {"L": "lora_head_r16_t6", "SL": "sequential_linear_t6", "FT": "sequential_ft_t6"}, {"L": "lora_head_r16_t6", "SL": "sequential_linear_t6", "FT": "sequential_ft_t6"}),
            ("T7 OASIS brain age", "MAE yr$\\downarrow$", 9, "mae", 2, 106, {"L": "lora_head_r16_t9", "SL": "sequential_linear_t9", "FT": "sequential_ft_t9"}, {"L": "lora_head_r16_t9", "SL": "sequential_linear_t9", "FT": "sequential_ft_t9"}),
            ("T8 OASIS sex", "AUC$\\uparrow$", 10, "auc", 3, 106, {"L": "lora_head_r16_t10", "SL": "sequential_linear_t10", "FT": "sequential_ft_t10"}, {"L": "lora_head_r16_t10", "SL": "sequential_linear_t10", "FT": "sequential_ft_t10"})):
        cells = []
        for k in ("L", "SL", "FT"):
            v42 = vals("n32", c32[k], task, metric, seeds=(42,))
            cells.append("n/a" if v42.size == 0 else f"{v42.mean():.{d}f}")
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
            k, n = const_count("n32", c, task, const_key, seeds=SW)
            return const_cell(v, d, k, n, best)
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
                     for _, c in strategies if vals("n32", f"{bb}{c}{suf}", task, metric, seeds=SW).size
                     and (ck is None or 2 * const_count("n32", f"{bb}{c}{suf}", task, ck, seeds=SW)[0]
                          < max(1, const_count("n32", f"{bb}{c}{suf}", task, ck, seeds=SW)[1]))}
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

    # ---- tab:controls: breaking the head-size / task-type confound ----------------
    # Rows: tasks. Columns: LoRA + head (reference), frozen backbone with only a small head
    # (segmentation: the 66-parameter output projection; global: the closed-form ridge /
    # logistic probe on z-scored features), frozen encoder with a larger head (segmentation:
    # the 8.45M decoder = Sequential Linear; global: a 2-layer MLP head), partial fine-tuning
    # (deepest encoder stage + everything downstream).
    ctrl = [
        ("T1 BraTS", 2, "dice", 3, None, {"L": "lora_head_r16", "SMALL": "head_only_t2", "LARGE": "sequential_linear", "PFT": "partial_ft_t2"}),
        ("T3 ISLES", 4, "dice", 3, None, {"L": "lora_head_r16_t4", "SMALL": "head_only_t4", "LARGE": "sequential_linear_t4", "PFT": "partial_ft_t4"}),
        ("T4 hippocampus", 7, "dice", 3, None, {"L": "lora_head_r16_t7", "SMALL": "head_only_t7", "LARGE": "sequential_linear_t7", "PFT": "partial_ft_t7"}),
        ("T6 WMH", 8, "dice", 3, None, {"L": "lora_head_r16_t8", "SMALL": "head_only_t8", "LARGE": "sequential_linear_t8", "PFT": "partial_ft_t8"}),
        ("T2 IXI age", 5, "mae", 2, "pred_std", {"L": "lora_head_r16_t5", "SMALL": "ridge_probe_t5", "LARGE": "mlp_head_t5", "PFT": "partial_ft_t5"}),
        ("T7 OASIS age", 9, "mae", 2, "pred_std", {"L": "lora_head_r16_t9", "SMALL": "ridge_probe_t9", "LARGE": "mlp_head_t9", "PFT": "partial_ft_t9"}),
        ("T5 IXI sex", 6, "auc", 3, "balanced_acc", {"L": "lora_head_r16_t6", "SMALL": "ridge_probe_t6", "LARGE": "mlp_head_t6", "PFT": "partial_ft_t6"}),
        ("T8 OASIS sex", 10, "auc", 3, "balanced_acc", {"L": "lora_head_r16_t10", "SMALL": "ridge_probe_t10", "LARGE": "mlp_head_t10", "PFT": "partial_ft_t10"}),
    ]
    L = []
    for name, task, metric, d, ck, cf in ctrl:
        cells = []
        for k in ("L", "SMALL", "LARGE", "PFT"):
            v = vals("n32", cf[k], task, metric)
            if v.size == 0:
                cells.append("n/a")
            elif ck is None:
                cells.append(ms(v, d))
            else:
                kk, nn = const_count("n32", cf[k], task, ck)
                cells.append(const_cell(v, d, kk, nn))
        L.append(f"{name:16s} & " + " & ".join(cells) + " \\\\")
        if task == 8:
            L.append("\\midrule")
    (TAB / "controls_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:lr: per-strategy learning rate, chosen on validation ------------------
    # Seeds 42-44 for every rate. The rate is chosen per strategy and task by the mean
    # validation metric of the kept epoch; the test metric at that rate is reported beside
    # the test metric at the shared rate 1e-3.
    LR = (("1e-4", "_lr1e-4"), ("3e-4", "_lr3e-4"), ("1e-3", ""))
    S3 = (42, 43, 44)
    def lr_cfg(c, task, suf):
        if suf == "":
            return {2: {"lora_head_r16_t2": "lora_head_r16", "sequential_linear_t2": "sequential_linear",
                        "sequential_ft_t2": "sequential_ft", "bitfit_t2": "bitfit", "adapter_t2": "adapter"}.get(c, c)}.get(task, c)
        return c + suf
    L = []
    for name, task, metric, d, lower in (("BraTS Dice", 2, "dice", 3, False), ("IXI age MAE", 5, "mae", 2, True), ("IXI sex AUC", 6, "auc", 3, False)):
        cells = []
        for lab, base in (("L", "lora_head_r16"), ("SL", "sequential_linear"), ("FT", "sequential_ft"), ("BF", "bitfit"), ("AD", "adapter")):
            c0 = f"{base}_t{task}"
            vmean = {}
            for lr, suf in LR:
                v = vals("n32", lr_cfg(c0, task, suf), task, metric, stage="val_selected", seeds=S3)
                if v.size == len(S3):
                    vmean[lr] = v.mean()
            if not vmean:
                cells += ["n/a", "n/a"]
                continue
            pick = (min if lower else max)(vmean, key=vmean.get)
            t_pick = vals("n32", lr_cfg(c0, task, dict(LR)[pick]), task, metric, seeds=S3)
            t_ref = vals("n32", lr_cfg(c0, task, ""), task, metric, seeds=S3)
            cells.append(f"{m(t_ref, d)}")
            cells.append(f"{m(t_pick, d)} ({pick})")
        L.append(f"{name:12s} & " + " & ".join(cells) + " \\\\")
    (TAB / "lr_body.tex").write_text("\n".join(L) + "\n")

    # ---- tab:forget25: BraTS -> whole-head IXI brain age (a learnable second task) ---
    L = []
    for lab, c in (("Sequential FT", "sequential_ft_t25"), ("EWC, $\\lambda{=}10^2$", "ewc_t25_ewc100"),
                   ("EWC, $\\lambda{=}10^3$", "ewc_t25"), ("EWC, $\\lambda{=}10^4$", "ewc_t25_ewc10000"),
                   ("LwF (feature distillation)", "lwf_t25"), ("LwF (output distillation)", "lwf_out_t25"),
                   ("Replay (feature distillation)", "replay_t25"), ("Replay (labelled rehearsal)", "replay_label_t25")):
        dice, age = vals("n32", c, 2, "dice"), vals("n32", c, 5, "mae")
        aft = vals("n32", c, 2, "dice", "test_after_task5")
        bwt = aft - dice[: aft.size] if aft.size else aft
        L.append(f"{lab:32s} & {ms(dice)} & {ms(age, 2)} & {ms(aft)} & {ms(bwt)} \\\\")
    (TAB / "forget25_body.tex").write_text("\n".join(L) + "\n")

    n_runs = len(glob.glob(str(OUT / "n*" / "*" / "seed*" / "results.json")))
    (TAB / "counts.tex").write_text(f"\\newcommand{{\\nruns}}{{{n_runs}}}\n")
    print("runs:", n_runs, "| tables written to", TAB)


if __name__ == "__main__":
    main()
