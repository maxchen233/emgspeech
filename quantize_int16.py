"""
INT16 量化工具 — 將 FP32 模型權重量化為 INT16 存儲

用法:
    python quantize_int16.py \
        --model_path models/transduction_model/model.pt \
        --output_path models/transduction_model/model_int16.pt \
        --hifigan_checkpoint hifigan_finetuned/checkpoint

原理:
    對每個 weight tensor 做 per-tensor affine 量化:
      scale = (max - min) / 65535
      zero_point = round(-min / scale)
      w_int16 = clamp(round(w / scale) + zero_point, 0, 65535)  → 存為 uint16
    反量化:
      w_fp32 ≈ (w_int16 - zero_point) * scale
"""

import os
import sys
import copy
import numpy as np

import torch
import torch.nn as nn

from architecture import Model
from data_utils import phoneme_inventory
from read_emg import EMGDataset
from vocoder import Vocoder  # 註冊 --hifigan_checkpoint flag

from absl import flags
FLAGS = flags.FLAGS

# 避免重複定義 flags (被其他腳本 import 時)
try:
    flags.DEFINE_string('model_path',
                        'models/transduction_model/model.pt',
                        '原始 FP32 模型路徑')
except flags.DuplicateFlagError:
    pass

try:
    flags.DEFINE_string('output_path',
                        'models/transduction_model/model_int16.pt',
                        '量化後 INT16 模型輸出路徑')
except flags.DuplicateFlagError:
    pass

try:
    flags.DEFINE_boolean('quantized', False, '是否使用 INT16 量化模型')
except flags.DuplicateFlagError:
    pass


# ═══════════════════════════════════════════════════════════════
# INT16 量化核心函式
# ═══════════════════════════════════════════════════════════════

def quantize_tensor_int16(tensor):
    """
    將 FP32 tensor 量化為 INT16 (unsigned, 0~65535)

    Args:
        tensor: FP32 torch.Tensor

    Returns:
        dict with keys:
            'data': int32 tensor (存 uint16 值, 範圍 0~65535)
            'scale': float32 scalar
            'zero_point': int scalar
            'shape': original shape
    """
    t = tensor.detach().float()
    t_min = t.min().item()
    t_max = t.max().item()

    # 避免 scale=0 (常數 tensor)
    if t_max == t_min:
        return {
            'data': torch.zeros(t.shape, dtype=torch.int32),
            'scale': torch.tensor(1.0, dtype=torch.float32),
            'zero_point': torch.tensor(0, dtype=torch.int32),
            'shape': t.shape,
        }

    qmin, qmax = 0, 65535
    scale = (t_max - t_min) / (qmax - qmin)
    zero_point = int(round(-t_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))

    # 量化
    q = torch.clamp(torch.round(t / scale) + zero_point, qmin, qmax)
    q = q.to(torch.int32)  # 用 int32 存 uint16 值

    return {
        'data': q,
        'scale': torch.tensor(scale, dtype=torch.float32),
        'zero_point': torch.tensor(zero_point, dtype=torch.int32),
        'shape': t.shape,
    }


def dequantize_tensor_int16(q_info):
    """
    將 INT16 量化 tensor 反量化回 FP32

    Args:
        q_info: quantize_tensor_int16() 的輸出 dict

    Returns:
        FP32 torch.Tensor
    """
    data = q_info['data'].float()
    scale = q_info['scale'].item()
    zero_point = q_info['zero_point'].item()
    return (data - zero_point) * scale


def quantize_state_dict_int16(state_dict):
    """
    量化整個 state_dict 中的 weight 參數

    只量化 .weight 結尾且 ndim >= 2 的 tensor (Linear, Conv1d 的權重)
    bias 和 BatchNorm 參數保持 FP32

    Returns:
        quantized_state_dict: dict
            - 量化的 key: '{name}.__q_data', '{name}.__q_scale', '{name}.__q_zp'
            - 未量化的 key: 保持原樣
        meta: 量化統計資訊
    """
    q_state = {}
    meta = {
        'quantized_keys': [],
        'skipped_keys': [],
        'total_fp32_bytes': 0,
        'total_int16_bytes': 0,
    }

    for key, tensor in state_dict.items():
        fp32_bytes = tensor.numel() * 4
        meta['total_fp32_bytes'] += fp32_bytes

        # 只量化 weight 且 ndim >= 2
        if 'weight' in key and tensor.ndim >= 2:
            q_info = quantize_tensor_int16(tensor)
            q_state[f'{key}.__q_data'] = q_info['data'].to(torch.int16)
            q_state[f'{key}.__q_scale'] = q_info['scale']
            q_state[f'{key}.__q_zp'] = q_info['zero_point']

            # int16 = 2 bytes + scale(4) + zp(4)
            int16_bytes = tensor.numel() * 2 + 8
            meta['total_int16_bytes'] += int16_bytes
            meta['quantized_keys'].append(key)

            # 計算量化誤差
            deq = dequantize_tensor_int16(q_info)
            err = (tensor.float() - deq).abs()
            rel_err = err.mean() / (tensor.float().abs().mean() + 1e-10)
            meta[f'error/{key}'] = {
                'mean_abs': err.mean().item(),
                'max_abs': err.max().item(),
                'relative': rel_err.item(),
            }
        else:
            q_state[key] = tensor
            meta['total_int16_bytes'] += fp32_bytes
            meta['skipped_keys'].append(key)

    return q_state, meta


def load_quantized_state_dict(q_state):
    """
    將量化的 state_dict 反量化回標準 FP32 state_dict

    Args:
        q_state: quantize_state_dict_int16() 輸出的 quantized state_dict

    Returns:
        標準 FP32 state_dict (可直接用 model.load_state_dict())
    """
    state_dict = {}
    processed = set()

    for key in q_state:
        if key.endswith('.__q_data'):
            base_key = key.replace('.__q_data', '')
            if base_key in processed:
                continue
            processed.add(base_key)

            q_info = {
                'data': q_state[f'{base_key}.__q_data'].to(torch.int32),
                'scale': q_state[f'{base_key}.__q_scale'],
                'zero_point': q_state[f'{base_key}.__q_zp'],
            }
            state_dict[base_key] = dequantize_tensor_int16(q_info)

        elif not key.endswith('.__q_scale') and not key.endswith('.__q_zp'):
            state_dict[key] = q_state[key]

    return state_dict


def save_quantized_model(state_dict, output_path):
    """量化並儲存模型"""
    q_state, meta = quantize_state_dict_int16(state_dict)

    save_data = {
        'quantized_state_dict': q_state,
        'quantization': 'int16',
        'meta': {
            'quantized_keys': meta['quantized_keys'],
            'skipped_keys': meta['skipped_keys'],
            'fp32_bytes': meta['total_fp32_bytes'],
            'int16_bytes': meta['total_int16_bytes'],
        }
    }

    torch.save(save_data, output_path)
    return meta


def load_quantized_model(model, checkpoint_path, device='cpu'):
    """
    載入 INT16 量化模型

    Args:
        model: 未初始化權重的 Model 實例
        checkpoint_path: INT16 量化模型路徑
        device: 目標裝置

    Returns:
        載入權重後的 model
    """
    data = torch.load(checkpoint_path, map_location=device)

    if isinstance(data, dict) and 'quantized_state_dict' in data:
        # INT16 量化格式
        q_state = data['quantized_state_dict']
        state_dict = load_quantized_state_dict(q_state)
        model.load_state_dict(state_dict)
        print(f'✅ 已載入 INT16 量化模型: {checkpoint_path}')
    else:
        # 標準 FP32 格式 (向後相容)
        if isinstance(data, dict) and 'state_dict' in data:
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
        print(f'⚠️ 這是 FP32 模型，非量化格式: {checkpoint_path}')

    return model


# ═══════════════════════════════════════════════════════════════
# 命令列入口
# ═══════════════════════════════════════════════════════════════

def main():
    model_path = FLAGS.model_path
    output_path = FLAGS.output_path

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 載入資料集 (只為取得模型維度) ──────────────────────────
    print('載入資料集 (取得模型維度)...')
    devset = EMGDataset(dev=True)
    n_phones = len(phoneme_inventory)

    # ── 載入 FP32 模型 ────────────────────────────────────────
    print(f'載入 FP32 模型: {model_path}')
    model = Model(devset.num_features, devset.num_speech_features,
                  n_phones).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── 量化 ──────────────────────────────────────────────────
    print('\n🔧 正在量化到 INT16...')
    meta = save_quantized_model(state_dict, output_path)

    # ── 報告 ──────────────────────────────────────────────────
    fp32_mb = meta['total_fp32_bytes'] / 1024 / 1024
    int16_mb = meta['total_int16_bytes'] / 1024 / 1024
    ratio = int16_mb / fp32_mb * 100

    print('\n' + '=' * 60)
    print('  INT16 量化報告')
    print('=' * 60)
    print(f'  原始大小 (FP32):  {fp32_mb:.1f} MB')
    print(f'  量化大小 (INT16): {int16_mb:.1f} MB')
    print(f'  壓縮比:           {ratio:.1f}%')
    print(f'  量化層數:         {len(meta["quantized_keys"])}')
    print(f'  跳過層數:         {len(meta["skipped_keys"])}')

    print(f'\n  量化的 layers:')
    for key in meta['quantized_keys']:
        err = meta.get(f'error/{key}', {})
        mean_err = err.get('mean_abs', 0)
        max_err = err.get('max_abs', 0)
        rel = err.get('relative', 0)
        print(f'    {key:45s} | 平均誤差={mean_err:.6f} | '
              f'最大誤差={max_err:.6f} | 相對誤差={rel:.6f}')

    print(f'\n  跳過的 layers (保持 FP32):')
    for key in meta['skipped_keys']:
        print(f'    {key}')

    # ── 驗證: 反量化後比較輸出 ────────────────────────────────
    print('\n🧪 驗證量化精度...')
    model_q = Model(devset.num_features, devset.num_speech_features,
                    n_phones).to(device)
    load_quantized_model(model_q, output_path, device)
    model_q.eval()

    # 用隨機輸入比較
    dummy_raw = torch.randn(1, 80, 8, device=device)
    dummy_feat = torch.randn(1, 10, devset.num_features, device=device)
    dummy_sess = torch.zeros(1, 10, dtype=torch.long, device=device)

    with torch.no_grad():
        out_fp32 = model(dummy_feat, dummy_raw, dummy_sess)
        out_int16 = model_q(dummy_feat, dummy_raw, dummy_sess)

    if isinstance(out_fp32, tuple):
        out_fp32 = out_fp32[0]
        out_int16 = out_int16[0]

    cos_sim = torch.nn.functional.cosine_similarity(
        out_fp32.flatten().unsqueeze(0),
        out_int16.flatten().unsqueeze(0)
    ).item()

    mse = (out_fp32 - out_int16).pow(2).mean().item()
    print(f'  Cosine Similarity: {cos_sim:.6f}')
    print(f'  MSE:               {mse:.8f}')

    # ── 檔案大小比較 ──────────────────────────────────────────
    fp32_file_mb = os.path.getsize(model_path) / 1024 / 1024
    int16_file_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'\n  原始檔案大小: {fp32_file_mb:.1f} MB')
    print(f'  量化檔案大小: {int16_file_mb:.1f} MB')
    print(f'  檔案壓縮比:   {int16_file_mb/fp32_file_mb*100:.1f}%')

    print(f'\n✅ 量化完成！輸出: {output_path}')
    print('=' * 60)


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
