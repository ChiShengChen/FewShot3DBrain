#!/usr/bin/env python3
"""
Stack a figure vertically N times.
Usage:
  python scripts/stack_figure.py outputs/figures/task2_seg_s42_0.png --n 6
  python scripts/stack_figure.py outputs/figures/task2_seg_s42_0.png --n 6 -o outputs/figures/task2_seg_s42_0_stack6.png
"""
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def stack_figure(img_path: str, n: int = 6, out_path: str = None) -> str:
    """Stack image vertically n times. Returns output path."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    stacked = np.tile(arr, (n, 1, 1))
    out = out_path or str(Path(img_path).with_stem(Path(img_path).stem + f"_stack{n}"))
    Image.fromarray(stacked).save(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input figure path")
    parser.add_argument("-n", "--num", type=int, default=6, help="Number of copies to stack")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: <stem>_stack<n>.png)")
    args = parser.parse_args()
    out = stack_figure(args.image, n=args.num, out_path=args.output)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
