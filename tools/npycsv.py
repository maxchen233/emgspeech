#!/usr/bin/env python3
"""Convert a NumPy .npy file to a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert_npy_to_csv(input_path: Path, output_path: Path) -> None:
    """Convert one .npy file to CSV without allowing pickle loading."""
    array = np.load(input_path, allow_pickle=False)

    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)

    np.savetxt(output_path, array, delimiter=",", fmt="%s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a NumPy .npy file to a CSV file."
    )
    parser.add_argument("input", type=Path, help="Input .npy file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output .csv file; defaults to the input filename with .csv extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_suffix(".csv")

    if input_path.suffix.lower() != ".npy":
        raise SystemExit(f"輸入檔案必須是 .npy：{input_path}")
    if not input_path.is_file():
        raise SystemExit(f"找不到輸入檔案：{input_path}")

    try:
        convert_npy_to_csv(input_path, output_path)
    except ValueError as exc:
        raise SystemExit(f"無法轉換檔案：{exc}") from exc

    print(f"已輸出：{output_path}")


if __name__ == "__main__":

    main()