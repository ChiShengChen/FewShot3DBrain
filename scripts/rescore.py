#!/usr/bin/env python3
"""
Re-score saved checkpoints under a different sliding-window setting, without retraining.

Needed whenever the evaluation protocol changes mid-flight: every method in a comparison
table has to be scored the same way, and a run that already finished under the old setting
would otherwise have to be thrown away and retrained.

  gq submit -g 5G -c 2 -m 12 -n rescore -- python scripts/rescore.py \
      --run_dir outputs/benchmark_v2/n32/lora/seed42 --overlap 0.25

Writes the updated numbers back into results.json and keeps the superseded ones under
"superseded_test" so the change is auditable.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS  # noqa: E402
from src.eval import evaluate_split  # noqa: E402
from src.splits import load_or_make_splits  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/run_benchmark_v2.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="outputs/benchmark_v2/n<shot>/<method>/seed<n>")
    ap.add_argument("--overlap", type=float, required=True)
    ap.add_argument("--data_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--split_dir", default=str(ROOT / "data/splits_v2"))
    ap.add_argument("--pretrained_path",
                    default=str(ROOT / "weights/fomo25_mmunetvae_pretrained.ckpt"))
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=64)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    res_path = run_dir / "results.json"
    if not res_path.exists():
        raise SystemExit(f"no results.json under {run_dir}")
    results = json.loads(res_path.read_text())

    old = results.get("eval_protocol", {}).get("overlap")
    if old is not None and abs(old - args.overlap) < 1e-9:
        print(f"already scored at overlap={args.overlap}; nothing to do")
        return

    # Backward-transfer entries need the post-sequence model state, which this script does
    # not reconstruct. Refuse up front rather than burn an evaluation pass and then leave
    # one row of the table scored at a different overlap from the rest.
    stale = {k: [f for f in e if f.startswith("test_after_task")]
             for k, e in results["per_task"].items()}
    stale = {k: v for k, v in stale.items() if v}
    if stale:
        raise SystemExit(
            f"{run_dir.name} carries backward-transfer results {stale} that this script "
            f"cannot re-score; re-run the method instead of mixing evaluation protocols.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = (args.patch_size,) * 3
    method, seed, n_shot = results["method"], results["seed"], results["n_shot"]
    print(f"re-scoring {method} seed{seed} n{n_shot}: overlap {old} -> {args.overlap}", flush=True)

    for key, entry in results["per_task"].items():
        task_id = int(key.replace("task", ""))
        cfg = TASK_CONFIGS[task_id]
        ckpt = run_dir / f"task{task_id}" / f"task{task_id}_best.pt"
        if not ckpt.exists():
            print(f"  {key}: no checkpoint, skipped", flush=True)
            continue

        splits = load_or_make_splits(args.data_dir, task_id, n_shot, seed=seed,
                                     split_dir=args.split_dir)
        model = _bench.build_model(method, task_id, cfg, patch, args.pretrained_path, args.lora_r)
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
        model = model.to(device)

        t0 = time.perf_counter()
        new = evaluate_split(model, splits["test"], task_id, device, patch=patch,
                             overlap=args.overlap, batch_size=args.eval_batch_size,
                             max_channels=cfg["modalities"], return_per_case=True)
        per_case = new.pop("per_case", None)
        if per_case is not None:
            (run_dir / f"task{task_id}_per_case.json").write_text(json.dumps(per_case, indent=2))

        entry.setdefault("superseded_test", {})[f"overlap_{old}"] = entry.get("test")
        entry["test"] = new
        print(f"  {key}: {new}  ({time.perf_counter() - t0:.0f}s)", flush=True)

        del model
        torch.cuda.empty_cache()

    results.setdefault("eval_protocol", {})["overlap"] = args.overlap
    results["eval_protocol"]["rescored_from"] = old
    res_path.write_text(json.dumps(results, indent=2))
    print(f"\nupdated {res_path}")


if __name__ == "__main__":
    main()
