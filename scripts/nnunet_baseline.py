#!/usr/bin/env python3
"""
nnU-Net v2 reference line: trained from scratch on the same 32 cases as every other method.

One nnU-Net model per (segmentation task, seed). The model trains on the seed's 32 "train"
cases, picks its checkpoint on the seed's "val" cases (nnU-Net's own validation fold, set
through a custom splits_final.json), and is scored on the seed's "test" cases with the
study's metrics (src.eval.dice_score / hd95, per case, in the study's working array space).
Results are filed where run_benchmark_v2.py files its runs:

    outputs/benchmark_v2/n32/nnunet_t<T>/seed<S>/results.json
    outputs/benchmark_v2/n32/nnunet_t<T>/seed<S>/task<T>_per_case.json

Stages (each a subcommand; `pipeline` chains train -> predict -> score for a gq job):

    convert   numpy case arrays -> nnU-Net raw dataset (imagesTr = train+val, imagesTs = test)
    plan      nnUNetv2_plan_and_preprocess (CPU) + custom splits_final.json
    prepare   convert + plan (the command a `gq --cpu-only` job runs)
    train     nnUNetv2_train <id> 3d_fullres 0 -tr nnUNetTrainer_250epochs --val_best
    predict   nnUNetv2_predict on imagesTs with checkpoint_best.pth
    score     Dice / HD95 against <case>_seg.npy -> results.json in the study's format
    pipeline  train && predict && score (the command a `gq -g 10G` job runs)
    describe  print the planned configuration (patch, batch, network) for the methods text

Environment: the isolated interpreter in .nnunet_env (python 3.11, torch 2.8.0+cu128, nnunetv2
2.5.1):
    conda create -p .nnunet_env python=3.11 -y
    .nnunet_env/bin/pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    .nnunet_env/bin/pip install nnunetv2==2.5.1
One patch inside the env (the upstream nnU-Net fix; original kept as polylr.py.orig): nnunetv2
2.5.1's PolyLRScheduler passes the `verbose` positional that torch >= 2.7 removed from
LRScheduler.__init__, so training dies with a TypeError at the first step without it:
    sed -i 's/else -1, False)/else -1)/' \
        .nnunet_env/lib/python3.11/site-packages/nnunetv2/training/lr_scheduler/polylr.py
Every GPU / long CPU stage must be submitted through `gq`; `submit-prepare` and
`submit-pipeline` do that (priority -1, ids recorded in outputs/nnunet_results/jobs.tsv).

Dataset id scheme: 500 + 10*task + (seed - 42)  ->  Dataset570_T7_s42, Dataset521_T2_s43, ...

Smoke test of the whole chain without a GPU (1 epoch on CPU, results to a scratch folder):
    gq run --cpu-only -c 6 -m 16 -p -1 -- .nnunet_env/bin/python scripts/nnunet_baseline.py pipeline \
        --task 7 --seed 42 --trainer nnUNetTrainer_1epoch --device cpu --out_root /tmp/nnunet_smoke
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENV_PY = ROOT / ".nnunet_env" / "bin" / "python"
ENV_BIN = ROOT / ".nnunet_env" / "bin"
RAW = ROOT / "data" / "nnunet_raw"
PRE = ROOT / "data" / "nnunet_preprocessed"
RES = ROOT / "outputs" / "nnunet_results"
BENCH = ROOT / "outputs" / "benchmark_v2" / "n32"
SPLIT_DIR = ROOT / "data" / "splits_v3"
LEDGER = RES / "jobs.tsv"

TASKS = {2: "Task002_FOMO2", 4: "Task004_ISLES_SEG", 7: "Task007_HIPPO", 8: "Task008_WMH_SEG"}
# Channel order as written by the project's prepare_* scripts.
CHANNELS = {2: ["T2", "FLAIR", "T1ce"], 4: ["DWI", "ADC", "FLAIR"], 7: ["T1"], 8: ["FLAIR", "T1"]}
N_SHOT = 32
CONFIG = "3d_fullres"
TRAINER = "nnUNetTrainer_250epochs"
FOLD = 0


def dataset_id(task, seed):
    return 500 + 10 * task + (seed - 42)


def dataset_name(task, seed):
    return f"Dataset{dataset_id(task, seed)}_T{task}_s{seed}"


def nnunet_env():
    env = dict(os.environ)
    env.update({"nnUNet_raw": str(RAW), "nnUNet_preprocessed": str(PRE), "nnUNet_results": str(RES),
                "nnUNet_n_proc_DA": env.get("nnUNet_n_proc_DA", "4"),
                # nnunetv2 2.5.1 calls torch.load without weights_only=False; torch >= 2.6 defaults
                # to weights_only=True and refuses nnU-Net's checkpoints (numpy scalars inside).
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                # torch.compile is a speed knob only (no effect on the model); it is untested with
                # this torch build and a compile failure would waste a queue slot, so it is off.
                "nnUNet_compile": env.get("nnUNet_compile", "f")})
    # The isolated env's console scripts come first so nnUNetv2_* resolves to 2.5.1.
    env["PATH"] = f"{ENV_BIN}:{env.get('PATH', '')}"
    return env


def load_split(task, seed):
    f = SPLIT_DIR / f"{TASKS[task]}_n{N_SHOT}_seed{seed}.json"
    s = json.load(open(f))["splits"]
    assert len(s["train"]) == N_SHOT, f"{f}: expected {N_SHOT} train cases, got {len(s['train'])}"
    assert not (set(s["train"]) | set(s["val"])) & set(s["test"]), f"{f}: test overlaps train/val"
    return s


def case_id(path):
    return os.path.basename(path)


def run(cmd, env=None, log=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([str(c) for c in cmd], env=env or nnunet_env())
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    print(f"  done in {time.perf_counter() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------
# convert
# ------------------------------------------------------------------------------------------

def _write_nii(arr, path, dtype):
    """Write a (X, Y, Z) array as NIfTI with identity geometry (1 mm, no rotation).

    SimpleITK is nnU-Net's default reader/writer for .nii.gz, so using it on both ends keeps
    the voxel grid identical to the numpy array: GetArrayFromImage(ReadImage(path)) == arr.
    """
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(dtype)))
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path), useCompression=True)


def _read_nii(path):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _convert_case(job):
    case, img_dir, lab_dir, n_ch = job
    cid = case_id(case)
    src = ROOT / case
    outs = [img_dir / f"{cid}_{c:04d}.nii.gz" for c in range(n_ch)]
    if lab_dir is not None:
        outs.append(lab_dir / f"{cid}.nii.gz")
    if all(o.exists() for o in outs):
        return cid, "skip"
    vol = np.load(str(src) + ".npy", allow_pickle=True)
    if vol.ndim == 3:
        vol = vol[None]
    assert vol.shape[0] >= n_ch, f"{case}: {vol.shape[0]} channels < {n_ch}"
    for c in range(n_ch):
        _write_nii(vol[c], img_dir / f"{cid}_{c:04d}.nii.gz", np.float32)
    if lab_dir is not None:
        seg = np.load(str(src) + "_seg.npy", allow_pickle=True)
        if seg.ndim == 4:
            seg = seg[0]
        assert seg.shape == vol.shape[1:], f"{case}: seg {seg.shape} vs vol {vol.shape[1:]}"
        _write_nii((seg > 0), lab_dir / f"{cid}.nii.gz", np.uint8)
    return cid, "ok"


def convert(task, seed, nproc=6):
    s = load_split(task, seed)
    name = dataset_name(task, seed)
    d = RAW / name
    img_tr, lab_tr, img_ts = d / "imagesTr", d / "labelsTr", d / "imagesTs"
    for p in (img_tr, lab_tr, img_ts):
        p.mkdir(parents=True, exist_ok=True)
    n_ch = len(CHANNELS[task])
    jobs = [(c, img_tr, lab_tr, n_ch) for c in s["train"] + s["val"]]
    jobs += [(c, img_ts, None, n_ch) for c in s["test"]]
    t0 = time.perf_counter()
    with Pool(nproc) as pool:
        for i, (cid, st) in enumerate(pool.imap_unordered(_convert_case, jobs), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {cid} {st}", flush=True)
    dataset_json = {
        "channel_names": {str(i): n for i, n in enumerate(CHANNELS[task])},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(s["train"]) + len(s["val"]),
        "file_ending": ".nii.gz",
        "name": name,
        "description": (f"FewShot3DBrain {TASKS[task]} n{N_SHOT} seed{seed}: imagesTr = the seed's "
                        f"{len(s['train'])} train + {len(s['val'])} val cases (val is nnU-Net's fold-0 "
                        f"validation set via splits_final.json); imagesTs = the {len(s['test'])} test cases. "
                        "Arrays are the study's preprocessed working space, written with identity 1 mm geometry."),
        "reference": "scripts/nnunet_baseline.py",
        "licence": "see source datasets",
        "release": "0.0",
    }
    (d / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    (d / "study_split.json").write_text(json.dumps({"task": task, "seed": seed, "n_shot": N_SHOT,
                                                    "split_file": f"data/splits_v3/{TASKS[task]}_n{N_SHOT}_seed{seed}.json"}, indent=2))
    print(f"converted {name}: {len(jobs)} cases in {time.perf_counter() - t0:.0f}s -> {d}", flush=True)


# ------------------------------------------------------------------------------------------
# plan / preprocess
# ------------------------------------------------------------------------------------------

def write_splits(task, seed):
    s = load_split(task, seed)
    pre_dir = PRE / dataset_name(task, seed)
    pre_dir.mkdir(parents=True, exist_ok=True)
    splits = [{"train": [case_id(c) for c in s["train"]], "val": [case_id(c) for c in s["val"]]}]
    (pre_dir / "splits_final.json").write_text(json.dumps(splits, indent=2))
    print(f"wrote {pre_dir / 'splits_final.json'} (fold 0: {len(splits[0]['train'])} train / "
          f"{len(splits[0]['val'])} val)", flush=True)


def plan(task, seed, nproc=6):
    did = dataset_id(task, seed)
    # Fingerprint, planning and preprocessing on imagesTr (train + val; test never enters).
    run(["nnUNetv2_plan_and_preprocess", "-d", did, "-c", CONFIG, "--verify_dataset_integrity",
         "-np", nproc, "-npfp", nproc])
    write_splits(task, seed)
    describe(task, seed)


def prepare(task, seed, nproc=6):
    convert(task, seed, nproc)
    plan(task, seed, nproc)


# ------------------------------------------------------------------------------------------
# train / predict
# ------------------------------------------------------------------------------------------

def fold_dir(task, seed, trainer=TRAINER):
    return RES / dataset_name(task, seed) / f"{trainer}__nnUNetPlans__{CONFIG}" / f"fold_{FOLD}"


def train(task, seed, trainer=TRAINER, device="cuda"):
    did = dataset_id(task, seed)
    fd = fold_dir(task, seed, trainer)
    if (fd / "checkpoint_final.pth").exists():
        print(f"{fd}: checkpoint_final.pth exists, training already complete", flush=True)
        return
    assert (PRE / dataset_name(task, seed) / "splits_final.json").exists(), "run `plan` first"
    cmd = ["nnUNetv2_train", did, CONFIG, FOLD, "-tr", trainer, "--val_best", "-device", device]
    if (fd / "checkpoint_latest.pth").exists():
        cmd.append("--c")  # resume after a kill (gq retry)
    run(cmd)


def pred_dir(task, seed, trainer=TRAINER):
    return RES / "predictions" / dataset_name(task, seed) / trainer


def predict(task, seed, trainer=TRAINER, device="cuda", checkpoint="checkpoint_best.pth"):
    did = dataset_id(task, seed)
    out = pred_dir(task, seed, trainer)
    out.mkdir(parents=True, exist_ok=True)
    s = load_split(task, seed)
    missing = [c for c in s["test"] if not (out / f"{case_id(c)}.nii.gz").exists()]
    if not missing:
        print(f"{out}: all {len(s['test'])} test predictions exist", flush=True)
        return
    assert (fold_dir(task, seed, trainer) / checkpoint).exists(), f"no {checkpoint} in {fold_dir(task, seed, trainer)}"
    # Default nnU-Net inference: sliding window at step 0.5 with Gaussian weighting and
    # mirroring test-time augmentation; -npp/-nps are the preprocessing/export workers.
    run(["nnUNetv2_predict", "-i", RAW / dataset_name(task, seed) / "imagesTs", "-o", out,
         "-d", did, "-c", CONFIG, "-f", FOLD, "-tr", trainer, "-chk", checkpoint,
         "-npp", 3, "-nps", 3, "-device", device, "--continue_prediction"])
    (out / "checkpoint_used.txt").write_text(checkpoint + "\n")


# ------------------------------------------------------------------------------------------
# score
# ------------------------------------------------------------------------------------------

def _plan_summary(task, seed):
    p = json.load(open(PRE / dataset_name(task, seed) / "nnUNetPlans.json"))
    c = p["configurations"][CONFIG]
    arch = c.get("architecture", {}).get("arch_kwargs", {})
    return {
        "patch_size": c["patch_size"], "batch_size": c["batch_size"], "spacing": c["spacing"],
        "n_stages": arch.get("n_stages"), "features_per_stage": arch.get("features_per_stage"),
        "strides": arch.get("strides"), "kernel_sizes": arch.get("kernel_sizes"),
        "n_conv_per_stage": arch.get("n_conv_per_stage"),
        "n_conv_per_stage_decoder": arch.get("n_conv_per_stage_decoder"),
        "network_class": c.get("architecture", {}).get("network_class_name"),
        "normalization_schemes": c.get("normalization_schemes"),
        "use_mask_for_norm": c.get("use_mask_for_norm"),
        "median_image_size_in_voxels": c.get("median_image_size_in_voxels"),
        "preprocessor": c.get("preprocessor_name"),
    }


def describe(task, seed):
    info = _plan_summary(task, seed)
    print(f"\n{dataset_name(task, seed)} {CONFIG} plan:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def _train_stats(fd):
    """Best EMA pseudo-Dice and epoch from checkpoint_best; wall time from the training logs."""
    import torch
    ck = torch.load(fd / "checkpoint_best.pth", map_location="cpu", weights_only=False)
    weights = ck["network_weights"]
    n_params = int(sum(int(np.prod(v.shape)) for v in weights.values()))
    best_ema = float(ck.get("_best_ema", float("nan")))
    best_epoch = int(ck.get("current_epoch", -1))
    ck_final = fd / "checkpoint_final.pth"
    n_epochs = None
    if ck_final.exists():
        n_epochs = int(torch.load(ck_final, map_location="cpu", weights_only=False).get("current_epoch", -1))
    epoch_sec = []
    for log in sorted(glob.glob(str(fd / "training_log_*.txt"))):
        for line in open(log, errors="replace"):
            m = re.search(r"Epoch time: ([\d.]+) s", line)
            if m:
                epoch_sec.append(float(m.group(1)))
    return {"n_params": n_params, "best_ema_pseudo_dice": best_ema, "best_epoch": best_epoch,
            "n_epochs_trained": n_epochs, "train_sec": float(sum(epoch_sec)), "n_epoch_times": len(epoch_sec)}


def _val_summary(fd):
    """Real validation-fold Dice (checkpoint_best thanks to --val_best), if nnU-Net wrote it."""
    f = fd / "validation" / "summary.json"
    if not f.exists():
        return None
    s = json.load(open(f))
    try:
        return {"dice": float(s["foreground_mean"]["Dice"]), "n": len(s["metric_per_case"])}
    except (KeyError, TypeError):
        return None


def score(task, seed, trainer=TRAINER, out_root=BENCH, pred_root=None, checkpoint="checkpoint_best.pth"):
    sys.path.insert(0, str(ROOT))
    from src.eval import dice_score, hd95  # noqa: E402  (the study's metric implementations)

    s = load_split(task, seed)
    pdir = Path(pred_root) if pred_root else pred_dir(task, seed, trainer)
    fd = fold_dir(task, seed, trainer)
    dices, hds, per_case = [], [], []
    for i, case in enumerate(s["test"], 1):
        pred_f = pdir / f"{case_id(case)}.nii.gz"
        if not pred_f.exists():
            raise SystemExit(f"missing prediction {pred_f} (run `predict` first)")
        pred = (_read_nii(pred_f) > 0).astype(np.uint8)
        gt = np.load(str(ROOT / case) + "_seg.npy", allow_pickle=True)
        if gt.ndim == 4:
            gt = gt[0]
        gt = (np.asarray(gt) > 0).astype(np.uint8)
        assert pred.shape == gt.shape, f"{case}: pred {pred.shape} != gt {gt.shape}"
        d, h = dice_score(pred, gt), hd95(pred, gt)
        dices.append(d)
        hds.append(h)
        per_case.append({"case": case, "dice": d, "hd95": h})
        if i % 25 == 0 or i == len(s["test"]):
            print(f"[{i}/{len(s['test'])}] running mean dice {np.mean(dices):.4f}", flush=True)
    hd_arr = np.array(hds, dtype=float)
    test = {
        "n": len(dices),
        "dice": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "dice_median": float(np.median(dices)),
        "hd95": float(np.nanmean(hd_arr)) if np.isfinite(hd_arr).any() else float("nan"),
        "hd95_undefined": int(np.isnan(hd_arr).sum()),
    }

    stats = _train_stats(fd)
    val_real = _val_summary(fd)
    plan_info = _plan_summary(task, seed)
    val_selected = {"dice": stats["best_ema_pseudo_dice"],
                    "ema_pseudo_dice": stats["best_ema_pseudo_dice"],
                    "epoch": stats["best_epoch"],
                    "n_params": stats["n_params"]}
    if val_real is not None:
        val_selected["val_fold_dice"] = val_real["dice"]
        val_selected["val_fold_n"] = val_real["n"]

    results = {
        "method": "nnunet",
        "seed": seed,
        "n_shot": N_SHOT,
        "tasks": [task],
        "eval_protocol": {
            "framework": "nnU-Net v2 (nnunetv2 2.5.1), trained from scratch on the seed's 32 train cases",
            "configuration": CONFIG,
            "trainer": trainer,
            "fold": FOLD,
            "checkpoint": checkpoint,
            "checkpoint_rule": "checkpoint_best = epoch with the best EMA validation pseudo-Dice on the seed's val cases",
            "inference": ("nnUNetv2_predict defaults: Gaussian-weighted sliding window at step 0.5 x patch, "
                          "mirroring test-time augmentation on all axes, argmax, prediction restored to the "
                          "full original array grid"),
            "space": "study working array space (same arrays as every other method), identity 1 mm geometry",
            "patch": plan_info["patch_size"],
            "batch_size": plan_info["batch_size"],
            "metrics": "src.eval.dice_score / hd95 per test case, binarised pred > 0 vs seg > 0",
            "max_test_subjects": None,
        },
        "per_task": {
            f"task{task}": {
                "position": 0,
                "val_selected": val_selected,
                "test": test,
                "params": {"trainable": stats["n_params"], "total": stats["n_params"], "trainable_pct": 100.0},
                "train_sec": stats["train_sec"],
                "n_epochs": stats["n_epochs_trained"],
                "n_train": len(s["train"]),
                "n_val": len(s["val"]),
                "n_test": len(s["test"]),
                "nnunet_plan": plan_info,
                "nnunet_dataset": dataset_name(task, seed),
            }
        },
    }
    out_dir = Path(out_root) / f"nnunet_t{task}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / f"task{task}_per_case.json").write_text(json.dumps(per_case, indent=2))
    print(f"task {task} seed {seed} TEST: {test}")
    print(f"wrote {out_dir / 'results.json'}")
    return results


def pipeline(task, seed, trainer=TRAINER, device="cuda", out_root=BENCH):
    train(task, seed, trainer, device)
    predict(task, seed, trainer, device)
    score(task, seed, trainer, out_root)


# ------------------------------------------------------------------------------------------
# gq submission helpers (print / record the job ids; the queue does the scheduling)
# ------------------------------------------------------------------------------------------

def _ledger(stage, task, seed, job_id, cmd):
    RES.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a") as f:
        if new:
            f.write("job_id\tstage\ttask\tseed\tsubmitted\tcommand\n")
        f.write(f"{job_id}\t{stage}\t{task}\t{seed}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\t{cmd}\n")


def _gq_submit(gq_args, cmd, stage, task, seed):
    full = ["gq", "submit", *gq_args, "--", *cmd]
    print("+", " ".join(map(str, full)), flush=True)
    out = subprocess.run([str(c) for c in full], capture_output=True, text=True, cwd=str(ROOT))
    print(out.stdout.strip(), out.stderr.strip())
    if out.returncode != 0:
        raise SystemExit("gq submit failed")
    m = re.search(r"(\d+)", out.stdout)
    jid = m.group(1) if m else "?"
    _ledger(stage, task, seed, jid, " ".join(map(str, full)))
    return jid


def submit_prepare(task, seed, priority=-1, expect="40m"):
    cmd = ["env", f"nnUNet_raw={RAW}", f"nnUNet_preprocessed={PRE}", f"nnUNet_results={RES}",
           ENV_PY, ROOT / "scripts" / "nnunet_baseline.py", "prepare", "--task", task, "--seed", seed, "--nproc", 6]
    return _gq_submit(["--cpu-only", "-c", 6, "-m", 24, "-p", priority, "-n", f"nnunet-prep-t{task}-s{seed}",
                       "-t", expect], cmd, "prepare", task, seed)


def submit_pipeline(task, seed, priority=-1, vram="10G", expect="9h"):
    cmd = ["env", f"nnUNet_raw={RAW}", f"nnUNet_preprocessed={PRE}", f"nnUNet_results={RES}",
           "nnUNet_n_proc_DA=4", ENV_PY, ROOT / "scripts" / "nnunet_baseline.py", "pipeline",
           "--task", task, "--seed", seed]
    return _gq_submit(["-g", vram, "-c", 6, "-m", 16, "-p", priority, "-n", f"nnunet-train-t{task}-s{seed}",
                       "-t", expect], cmd, "pipeline", task, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["convert", "plan", "prepare", "train", "predict", "score", "pipeline",
                                      "describe", "submit-prepare", "submit-pipeline"])
    ap.add_argument("--task", type=int, required=True, choices=sorted(TASKS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--trainer", default=TRAINER)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--out_root", default=str(BENCH), help="score: where nnunet_t<T>/seed<S>/ goes")
    ap.add_argument("--pred_root", default=None, help="score: prediction folder override")
    ap.add_argument("--priority", type=int, default=-1)
    ap.add_argument("--vram", default="10G")
    ap.add_argument("--expect", default=None)
    a = ap.parse_args()

    if a.stage == "convert":
        convert(a.task, a.seed, a.nproc)
    elif a.stage == "plan":
        plan(a.task, a.seed, a.nproc)
    elif a.stage == "prepare":
        prepare(a.task, a.seed, a.nproc)
    elif a.stage == "train":
        train(a.task, a.seed, a.trainer, a.device)
    elif a.stage == "predict":
        predict(a.task, a.seed, a.trainer, a.device, a.checkpoint)
    elif a.stage == "score":
        score(a.task, a.seed, a.trainer, a.out_root, a.pred_root, a.checkpoint)
    elif a.stage == "pipeline":
        pipeline(a.task, a.seed, a.trainer, a.device, a.out_root)
    elif a.stage == "describe":
        describe(a.task, a.seed)
    elif a.stage == "submit-prepare":
        submit_prepare(a.task, a.seed, a.priority, a.expect or "40m")
    elif a.stage == "submit-pipeline":
        submit_pipeline(a.task, a.seed, a.priority, a.vram, a.expect or "9h")


if __name__ == "__main__":
    main()
