#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge a structurally-pruned / fine-tuned CNN frontend back into the original
CNN-Mamba checkpoint.

Input
-----
1) Original full CNN-Mamba checkpoint, e.g. model.pt
2) CNN fine-tuned checkpoint produced by cnn_only_network_slimming_standalone.py,
   e.g. cnn_slimming_output/cnn_finetuned_best.pt

Output
------
1) merged_model.pt
   - Keeps ALL original non-CNN tensors (Mamba, LayerNorm, heads, etc.)
   - Replaces only:
       conv_blocks.*
       w_raw_in.*
     with the fine-tuned/pruned CNN tensors.
   - Preserves the original checkpoint wrapper format when possible.

2) merged_model_prune_config.json
   - Stores model_size and pruned mid_channels needed to rebuild the CNN shape.

Important
---------
Because Network Slimming changed the internal widths of the CNN ResBlocks,
the old fixed-width architecture.py cannot strictly load merged_model.pt.
Your full model architecture must construct the three CNN blocks using the
saved mid_channels values while keeping every block output at model_size.
The Mamba side itself does NOT need to be pruned or reshaped.
"""

import argparse
import copy
import json
from pathlib import Path

import torch


CNN_PREFIXES = ("conv_blocks.", "w_raw_in.")


def is_tensor_state_dict(obj):
    return (
        isinstance(obj, dict)
        and len(obj) > 0
        and all(isinstance(k, str) for k in obj.keys())
        and any(torch.is_tensor(v) for v in obj.values())
    )


def find_state_dict(checkpoint):
    """
    Return:
        state_dict, wrapper_key

    wrapper_key is None when checkpoint itself is a state_dict.
    """
    if is_tensor_state_dict(checkpoint):
        return checkpoint, None

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported full checkpoint type: {type(checkpoint).__name__}"
        )

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if is_tensor_state_dict(value):
            return value, key

    raise ValueError(
        "Cannot find full model state_dict. Expected either a plain state_dict "
        "or a checkpoint containing state_dict/model_state_dict/model."
    )


def strip_module_prefix(state):
    """
    Normalize DataParallel/DDP state_dict keys to names without 'module.'.
    Returns (normalized_state, had_module_prefix).
    """
    if not state:
        return state, False

    tensor_keys = [k for k, v in state.items() if torch.is_tensor(v)]
    had_module = bool(tensor_keys) and all(k.startswith("module.") for k in tensor_keys)

    if not had_module:
        return dict(state), False

    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out, True


def restore_module_prefix(state, use_module_prefix):
    if not use_module_prefix:
        return state
    return {
        (k if k.startswith("module.") else "module." + k): v
        for k, v in state.items()
    }


def load_finetuned_cnn(path):
    obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        raise ValueError("Fine-tuned CNN checkpoint must be a dict.")

    if "cnn_state_dict" not in obj:
        raise KeyError(
            "Fine-tuned CNN checkpoint does not contain 'cnn_state_dict'. "
            "Please use cnn_finetuned_best.pt produced by your slimming notebook."
        )

    state = obj["cnn_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("'cnn_state_dict' is not a dict.")

    state, _ = strip_module_prefix(state)

    model_size = obj.get("model_size")
    mid_channels = obj.get("mid_channels")

    if model_size is None:
        if "w_raw_in.weight" not in state:
            raise KeyError("Cannot infer model_size: missing w_raw_in.weight")
        model_size = int(state["w_raw_in.weight"].shape[0])
    else:
        model_size = int(model_size)

    if mid_channels is None:
        mids = []
        i = 0
        while f"conv_blocks.{i}.conv1.weight" in state:
            mids.append(int(state[f"conv_blocks.{i}.conv1.weight"].shape[0]))
            i += 1
        mid_channels = mids
    else:
        mid_channels = [int(x) for x in mid_channels]

    if len(mid_channels) != 3:
        raise ValueError(
            f"Expected 3 pruned CNN blocks, got mid_channels={mid_channels}"
        )

    return state, obj, model_size, mid_channels


def is_cnn_key(key):
    return key.startswith(CNN_PREFIXES)


def clone_tensor_or_value(v):
    if torch.is_tensor(v):
        return v.detach().cpu().clone()
    return copy.deepcopy(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original",
        required=True,
        help="Original full CNN-Mamba checkpoint, e.g. model.pt",
    )
    parser.add_argument(
        "--cnn",
        required=True,
        help="Fine-tuned CNN checkpoint, e.g. cnn_finetuned_best.pt",
    )
    parser.add_argument(
        "--output",
        default="merged_model.pt",
        help="Output merged full CNN-Mamba checkpoint",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Output prune config JSON. Default: <output_stem>_prune_config.json",
    )
    args = parser.parse_args()

    original_obj = torch.load(args.original, map_location="cpu")
    original_state_raw, wrapper_key = find_state_dict(original_obj)
    original_state, original_had_module = strip_module_prefix(original_state_raw)

    cnn_state, cnn_meta, model_size, mid_channels = load_finetuned_cnn(args.cnn)

    # ------------------------------------------------------------------
    # Validate original full checkpoint
    # ------------------------------------------------------------------
    original_cnn_keys = {k for k in original_state if is_cnn_key(k)}
    finetuned_cnn_keys = {k for k in cnn_state if is_cnn_key(k)}

    if not original_cnn_keys:
        raise KeyError(
            "Original full checkpoint has no conv_blocks.* / w_raw_in.* keys."
        )

    if not finetuned_cnn_keys:
        raise KeyError(
            "Fine-tuned CNN checkpoint has no conv_blocks.* / w_raw_in.* keys."
        )

    # Fine-tuned bundle is expected to contain all CNN tensors that existed
    # in the original model. The tensor SHAPES are allowed to differ because
    # structural pruning changes internal CNN widths.
    missing_from_finetuned = sorted(original_cnn_keys - finetuned_cnn_keys)
    if missing_from_finetuned:
        raise KeyError(
            "Fine-tuned CNN is missing CNN keys required by the original model:\n"
            + "\n".join(missing_from_finetuned[:50])
        )

    # Validate fixed 512-D bridge / model-size interface.
    if "w_raw_in.weight" not in cnn_state:
        raise KeyError("Fine-tuned CNN is missing w_raw_in.weight")

    w = cnn_state["w_raw_in.weight"]
    if tuple(w.shape) != (model_size, model_size):
        raise ValueError(
            f"w_raw_in.weight shape must stay [{model_size}, {model_size}], "
            f"got {tuple(w.shape)}"
        )

    # Each ResBlock output must remain model_size:
    # conv2: [model_size, pruned_mid, 3]
    for i, mid in enumerate(mid_channels):
        k1 = f"conv_blocks.{i}.conv1.weight"
        k2 = f"conv_blocks.{i}.conv2.weight"

        if k1 not in cnn_state or k2 not in cnn_state:
            raise KeyError(f"Missing {k1} or {k2}")

        conv1 = cnn_state[k1]
        conv2 = cnn_state[k2]

        if int(conv1.shape[0]) != mid:
            raise ValueError(
                f"Block {i}: conv1 output={conv1.shape[0]} but mid_channels={mid}"
            )
        if int(conv2.shape[0]) != model_size:
            raise ValueError(
                f"Block {i}: conv2 output must remain model_size={model_size}, "
                f"got {conv2.shape[0]}"
            )
        if int(conv2.shape[1]) != mid:
            raise ValueError(
                f"Block {i}: conv2 input={conv2.shape[1]} but mid_channels={mid}"
            )

    # ------------------------------------------------------------------
    # Merge: original full checkpoint + fine-tuned CNN
    # ------------------------------------------------------------------
    merged = {k: clone_tensor_or_value(v) for k, v in original_state.items()}

    replaced = []
    shape_changes = []

    for k in sorted(finetuned_cnn_keys):
        new_v = cnn_state[k]

        old_shape = (
            tuple(original_state[k].shape)
            if k in original_state and torch.is_tensor(original_state[k])
            else None
        )
        new_shape = tuple(new_v.shape) if torch.is_tensor(new_v) else None

        if old_shape != new_shape:
            shape_changes.append((k, old_shape, new_shape))

        merged[k] = clone_tensor_or_value(new_v)
        replaced.append(k)

    # ------------------------------------------------------------------
    # Critical check: every NON-CNN tensor must remain bit-identical
    # ------------------------------------------------------------------
    changed_non_cnn = []
    for k, old_v in original_state.items():
        if is_cnn_key(k):
            continue
        new_v = merged[k]
        if torch.is_tensor(old_v):
            if not torch.equal(old_v.cpu(), new_v.cpu()):
                changed_non_cnn.append(k)
        else:
            if old_v != new_v:
                changed_non_cnn.append(k)

    if changed_non_cnn:
        raise RuntimeError(
            "BUG: non-CNN tensors changed during merge:\n"
            + "\n".join(changed_non_cnn[:50])
        )

    # Restore original DataParallel 'module.' style if the source used it.
    merged_to_save = restore_module_prefix(merged, original_had_module)

    # Preserve the original wrapper structure.
    if wrapper_key is None:
        output_obj = merged_to_save
    else:
        output_obj = copy.deepcopy(original_obj)
        output_obj[wrapper_key] = merged_to_save

        # Optional metadata does not interfere with the model state_dict.
        output_obj["cnn_slimming"] = {
            "format": "cnn_mamba_merged_v1",
            "model_size": model_size,
            "mid_channels": mid_channels,
            "cnn_source": str(args.cnn),
            "original_full_model": str(args.original),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_obj, output_path)

    if args.config is None:
        config_path = output_path.with_name(
            output_path.stem + "_prune_config.json"
        )
    else:
        config_path = Path(args.config)

    config = {
        "format": "cnn_mamba_merged_v1",
        "model_size": model_size,
        "mid_channels": mid_channels,
        "original_full_checkpoint": str(args.original),
        "finetuned_cnn_checkpoint": str(args.cnn),
        "merged_checkpoint": str(output_path),
        "cnn_prefixes_replaced": list(CNN_PREFIXES),
        "mamba_and_heads_unchanged": True,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    non_cnn_tensor_count = sum(
        1 for k, v in merged.items()
        if (not is_cnn_key(k)) and torch.is_tensor(v)
    )

    print("=" * 78)
    print("CNN-Mamba merge completed")
    print("=" * 78)
    print(f"Original full model : {args.original}")
    print(f"Fine-tuned CNN      : {args.cnn}")
    print(f"Output model        : {output_path}")
    print(f"Prune config        : {config_path}")
    print()
    print(f"model_size          : {model_size}")
    print(f"mid_channels        : {mid_channels}")
    print(f"CNN tensors replaced: {len(replaced)}")
    print(f"Non-CNN tensors kept: {non_cnn_tensor_count}")
    print(f"Non-CNN changed     : {len(changed_non_cnn)}")
    print()
    print("CNN shape changes:")
    for k, old_shape, new_shape in shape_changes:
        print(f"  {k}: {old_shape} -> {new_shape}")

    print()
    print("[OK] Mamba / LayerNorm / output-head tensors are unchanged.")
    print("[IMPORTANT] architecture.py must rebuild CNN blocks with:")
    print(f"            mid_channels={mid_channels}")
    print(f"            model_size={model_size}")
    print("            while each block output remains model_size.")


if __name__ == "__main__":
    main()