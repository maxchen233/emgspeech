#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export structurally-pruned CNN + Mamba checkpoint to the raw float32 .bin
format used by the existing C++ WeightMap / EmgModel implementation.

This exporter does NOT need to import architecture.py or mamba_ssm.
It reads tensors directly from the checkpoint, fuses BatchNorm into the
CNN convolutions, and writes:

    output_dir/
        model.json
        res0_conv1_weight.bin
        res0_conv1_bias.bin
        ...
        mamba0_*.bin
        ...
        final_norm_*.bin
        w_out_*.bin

It supports pruned CNN widths automatically, e.g.
    [136, 320, 328]

The public interface between CNN and Mamba remains d_model=512.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def looks_like_state_dict(obj):
    return (
        isinstance(obj, dict)
        and any(torch.is_tensor(v) for v in obj.values())
    )


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}"
        )

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if looks_like_state_dict(value):
            checkpoint = value
            break

    if not looks_like_state_dict(checkpoint):
        raise ValueError(
            "Cannot find model state_dict. Expected a plain state_dict "
            "or a checkpoint containing state_dict/model_state_dict/model."
        )

    out = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue

        # DataParallel / DDP
        while key.startswith("module."):
            key = key[len("module."):]

        # torch.compile can occasionally add this prefix
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]

        out[key] = value.detach().cpu()

    return out


def require(state, key):
    if key not in state:
        candidates = [
            k for k in state
            if k.endswith(key) or key in k
        ][:20]
        extra = ""
        if candidates:
            extra = "\nPossible related keys:\n  " + "\n  ".join(candidates)

        raise KeyError(
            f"Checkpoint is missing required tensor: {key}{extra}"
        )
    return state[key]


def optional(state, key):
    return state.get(key)


def as_f32_numpy(t):
    return (
        t.detach()
        .cpu()
        .contiguous()
        .to(torch.float32)
        .numpy()
    )


def fuse_conv_bn(
    conv_w,
    conv_b,
    bn_w,
    bn_b,
    running_mean,
    running_var,
    eps=1e-5,
):
    """
    Fuse PyTorch Conv1d + BatchNorm1d in eval mode.

        y = BN(conv(x))

    into:

        y = conv_fused(x)
    """
    w = conv_w.detach().cpu().float()
    out_channels = w.shape[0]

    if conv_b is None:
        b = torch.zeros(out_channels, dtype=torch.float32)
    else:
        b = conv_b.detach().cpu().float()

    gamma = bn_w.detach().cpu().float()
    beta = bn_b.detach().cpu().float()
    mean = running_mean.detach().cpu().float()
    var = running_var.detach().cpu().float()

    scale = gamma / torch.sqrt(var + eps)

    view_shape = [out_channels] + [1] * (w.ndim - 1)
    fused_w = w * scale.reshape(view_shape)
    fused_b = (b - mean) * scale + beta

    return fused_w.contiguous(), fused_b.contiguous()


class CppExporter:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.tensors = {}
        self.file_index = 0

    @staticmethod
    def safe_filename(name):
        return (
            name.replace(".", "_")
            .replace("/", "_")
            .replace("\\", "_")
            + ".bin"
        )

    def add_tensor(self, name, tensor):
        arr = as_f32_numpy(tensor)

        filename = self.safe_filename(name)
        path = self.out_dir / filename

        arr.tofile(path)

        self.tensors[name] = {
            "file": filename,
            "shape": list(arr.shape),
            "numel": int(arr.size),
            "dtype": "float32",
        }

    def add_bool(self, name, value):
        self.tensors[name] = bool(value)


def count_mamba_layers(state):
    indices = set()

    prefix = "mamba_layers."
    for key in state:
        if not key.startswith(prefix):
            continue

        rest = key[len(prefix):]
        first = rest.split(".", 1)[0]

        if first.isdigit():
            indices.add(int(first))

    if not indices:
        raise ValueError(
            "No mamba_layers.* tensors found in checkpoint."
        )

    n = max(indices) + 1

    expected = set(range(n))
    if indices != expected:
        raise ValueError(
            f"Mamba layer indices are not contiguous: {sorted(indices)}"
        )

    return n


def validate_cnn(state, d_model):
    mids = []

    for i in range(3):
        p = f"conv_blocks.{i}"

        w1 = require(state, f"{p}.conv1.weight")
        w2 = require(state, f"{p}.conv2.weight")

        mid = int(w1.shape[0])
        mids.append(mid)

        expected_in = 8 if i == 0 else d_model

        if list(w1.shape[1:]) != [expected_in, 3]:
            raise ValueError(
                f"Block {i} conv1 shape invalid: {tuple(w1.shape)}; "
                f"expected ({mid}, {expected_in}, 3)"
            )

        if tuple(w2.shape) != (d_model, mid, 3):
            raise ValueError(
                f"Block {i} conv2 shape invalid: {tuple(w2.shape)}; "
                f"expected ({d_model}, {mid}, 3)"
            )

        # BN dimensions
        if require(state, f"{p}.bn1.weight").numel() != mid:
            raise ValueError(f"Block {i} bn1 does not match pruned width {mid}")

        if require(state, f"{p}.bn2.weight").numel() != d_model:
            raise ValueError(f"Block {i} bn2 does not match d_model {d_model}")

    return mids


def export_resblock(state, ex, idx, d_model, bn_eps):
    p = f"conv_blocks.{idx}"

    # conv1 + bn1
    conv1_w, conv1_b = fuse_conv_bn(
        require(state, f"{p}.conv1.weight"),
        optional(state, f"{p}.conv1.bias"),
        require(state, f"{p}.bn1.weight"),
        require(state, f"{p}.bn1.bias"),
        require(state, f"{p}.bn1.running_mean"),
        require(state, f"{p}.bn1.running_var"),
        eps=bn_eps,
    )

    # conv2 + bn2
    conv2_w, conv2_b = fuse_conv_bn(
        require(state, f"{p}.conv2.weight"),
        optional(state, f"{p}.conv2.bias"),
        require(state, f"{p}.bn2.weight"),
        require(state, f"{p}.bn2.bias"),
        require(state, f"{p}.bn2.running_mean"),
        require(state, f"{p}.bn2.running_var"),
        eps=bn_eps,
    )

    ex.add_tensor(f"res{idx}.conv1.weight", conv1_w)
    ex.add_tensor(f"res{idx}.conv1.bias", conv1_b)
    ex.add_tensor(f"res{idx}.conv2.weight", conv2_w)
    ex.add_tensor(f"res{idx}.conv2.bias", conv2_b)

    skip_w_key = f"{p}.residual_path.weight"
    has_skip = skip_w_key in state
    ex.add_bool(f"res{idx}.has_skip", has_skip)

    if has_skip:
        skip_w, skip_b = fuse_conv_bn(
            require(state, skip_w_key),
            optional(state, f"{p}.residual_path.bias"),
            require(state, f"{p}.res_norm.weight"),
            require(state, f"{p}.res_norm.bias"),
            require(state, f"{p}.res_norm.running_mean"),
            require(state, f"{p}.res_norm.running_var"),
            eps=bn_eps,
        )

        ex.add_tensor(f"res{idx}.skip.weight", skip_w)
        ex.add_tensor(f"res{idx}.skip.bias", skip_b)


def export_mamba_layer(state, ex, i):
    src = f"mamba_layers.{i}"
    dst = f"mamba{i}"

    # Outer MambaBlock LayerNorm
    ex.add_tensor(
        f"{dst}.norm.weight",
        require(state, f"{src}.norm.weight"),
    )
    ex.add_tensor(
        f"{dst}.norm.bias",
        require(state, f"{src}.norm.bias"),
    )

    # mamba_ssm Mamba internals
    mp = f"{src}.mamba"

    mapping = {
        "in_proj.weight": "in_proj.weight",
        "conv1d.weight": "conv1d.weight",
        "conv1d.bias": "conv1d.bias",
        "x_proj.weight": "x_proj.weight",
        "dt_proj.weight": "dt_proj.weight",
        "dt_proj.bias": "dt_proj.bias",
        "A_log": "A_log",
        "D": "D",
        "out_proj.weight": "out_proj.weight",
    }

    for src_suffix, dst_suffix in mapping.items():
        ex.add_tensor(
            f"{dst}.{dst_suffix}",
            require(state, f"{mp}.{src_suffix}"),
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Merged pruned CNN-Mamba .pt checkpoint",
    )

    ap.add_argument(
        "--output_dir",
        default="exported_weights",
        help="Directory used by C++ WeightMap",
    )

    ap.add_argument(
        "--bn_eps",
        type=float,
        default=1e-5,
        help="BatchNorm epsilon used by PyTorch BatchNorm1d",
    )

    ap.add_argument(
        "--clean",
        action="store_true",
        help="Delete output_dir before exporting. "
             "Do NOT use this if the same directory already contains HiFi-GAN weights.",
    )

    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Export pruned CNN-Mamba -> C++ float32 BIN")
    print("=" * 78)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output dir : {out_dir}")

    ckpt = torch_load_cpu(checkpoint_path)
    state = extract_state_dict(ckpt)

    # ----------------------------------------------------------
    # Infer model dimensions directly from checkpoint
    # ----------------------------------------------------------
    w_raw = require(state, "w_raw_in.weight")

    if w_raw.ndim != 2 or w_raw.shape[0] != w_raw.shape[1]:
        raise ValueError(
            f"w_raw_in.weight should be square, got {tuple(w_raw.shape)}"
        )

    d_model = int(w_raw.shape[0])
    mid_channels = validate_cnn(state, d_model)

    num_layers = count_mamba_layers(state)

    # Infer Mamba dimensions from layer 0.
    A_log = require(state, "mamba_layers.0.mamba.A_log")
    conv1d_w = require(state, "mamba_layers.0.mamba.conv1d.weight")
    in_proj_w = require(state, "mamba_layers.0.mamba.in_proj.weight")
    x_proj_w = require(state, "mamba_layers.0.mamba.x_proj.weight")

    if A_log.ndim != 2:
        raise ValueError(
            f"A_log must be 2-D, got {tuple(A_log.shape)}"
        )

    d_inner = int(A_log.shape[0])
    d_state = int(A_log.shape[1])

    if conv1d_w.ndim != 3:
        raise ValueError(
            f"Mamba conv1d.weight must be 3-D, got {tuple(conv1d_w.shape)}"
        )

    d_conv = int(conv1d_w.shape[2])

    if int(in_proj_w.shape[0]) != 2 * d_inner:
        raise ValueError(
            f"in_proj.weight shape {tuple(in_proj_w.shape)} "
            f"is inconsistent with d_inner={d_inner}"
        )

    dt_rank = int(x_proj_w.shape[0]) - 2 * d_state
    if dt_rank <= 0:
        raise ValueError(
            f"Invalid dt_rank inferred from x_proj.weight: {dt_rank}"
        )

    w_out = require(state, "w_out.weight")
    num_outs = int(w_out.shape[0])

    has_aux = "w_aux.weight" in state
    num_aux_outs = (
        int(state["w_aux.weight"].shape[0])
        if has_aux
        else 0
    )

    print()
    print("Detected architecture:")
    print(f"  d_model      = {d_model}")
    print(f"  mid_channels = {mid_channels}")
    print(f"  num_layers   = {num_layers}")
    print(f"  d_inner      = {d_inner}")
    print(f"  d_state      = {d_state}")
    print(f"  d_conv       = {d_conv}")
    print(f"  dt_rank      = {dt_rank}")
    print(f"  num_outs     = {num_outs}")
    print(f"  num_aux_outs = {num_aux_outs}")

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------
    ex = CppExporter(out_dir)

    for i in range(3):
        export_resblock(
            state,
            ex,
            i,
            d_model,
            args.bn_eps,
        )

    ex.add_tensor(
        "w_raw_in.weight",
        require(state, "w_raw_in.weight"),
    )
    ex.add_tensor(
        "w_raw_in.bias",
        require(state, "w_raw_in.bias"),
    )

    for i in range(num_layers):
        export_mamba_layer(
            state,
            ex,
            i,
        )

    ex.add_tensor(
        "final_norm.weight",
        require(state, "final_norm.weight"),
    )
    ex.add_tensor(
        "final_norm.bias",
        require(state, "final_norm.bias"),
    )

    ex.add_tensor(
        "w_out.weight",
        require(state, "w_out.weight"),
    )
    ex.add_tensor(
        "w_out.bias",
        require(state, "w_out.bias"),
    )

    if has_aux:
        ex.add_tensor(
            "w_aux.weight",
            require(state, "w_aux.weight"),
        )
        ex.add_tensor(
            "w_aux.bias",
            require(state, "w_aux.bias"),
        )

    manifest = {
        "format": "cnn_mamba_cpp_float32_v2",
        "checkpoint": str(checkpoint_path),
        "d_model": d_model,
        "num_layers": num_layers,
        "d_state": d_state,
        "d_conv": d_conv,
        "d_inner": d_inner,
        "dt_rank": dt_rank,
        "num_outs": num_outs,
        "num_aux_outs": num_aux_outs,
        "cnn_mid_channels": mid_channels,
        "cnn_downsample": 8,
        "bn_fused": True,
        "bn_eps": float(args.bn_eps),
        "tensors": ex.tensors,
    }

    model_json = out_dir / "model.json"
    model_json.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Small human-readable report.
    report_lines = [
        "CNN-Mamba C++ export",
        f"checkpoint={checkpoint_path}",
        f"d_model={d_model}",
        f"mid_channels={mid_channels}",
        f"num_layers={num_layers}",
        f"d_inner={d_inner}",
        f"d_state={d_state}",
        f"d_conv={d_conv}",
        f"dt_rank={dt_rank}",
        f"num_outs={num_outs}",
        f"num_aux_outs={num_aux_outs}",
        "",
        "Exported tensor shapes:",
    ]

    for name, info in ex.tensors.items():
        if isinstance(info, dict):
            report_lines.append(
                f"{name:32s} {info['shape']}"
            )
        else:
            report_lines.append(
                f"{name:32s} {info}"
            )

    (out_dir / "emg_export_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print(f"[OK] Wrote {len(ex.tensors)} entries")
    print(f"[OK] {model_json}")
    print(f"[OK] {out_dir / 'emg_export_report.txt'}")
    print()
    print("Important:")
    print("  CNN BatchNorm has been fused into Conv1d weights/biases.")
    print("  The C++ engine therefore does not need BatchNorm operators.")
    print("  Existing HiFi-GAN / normalizer files in output_dir were not touched.")


if __name__ == "__main__":
    main()
