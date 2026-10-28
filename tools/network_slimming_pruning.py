#!/usr/bin/env python3
"""
🔪 Network Slimming — CNN+Mamba 轉導模型剪枝 (CLI 版)

將 network_slimming_pruning.ipynb 轉為可在 WSL 終端直接跑的腳本。

流程:
  1. 載入預訓練模型，分析 BatchNorm γ 分佈
  2. Sparsity-aware 重新訓練 (BN γ L1 正則化)
  3. 根據 BN γ 閾值剪枝通道，建立瘦身後的模型
  4. Fine-tune 剪枝後的模型
  5. 匯出剪枝模型與通道配置
  6. 推論速度基準測試

用法:
    python network_slimming_pruning.py \\
        --model_path models_mamba1/transduction_model/model.pt \\
        --output_dir pruned_output \\
        --prune_percent 0.5

    # 使用真實資料:
    python network_slimming_pruning.py \\
        --model_path model.pt \\
        --data_dir emg_data \\
        --use_real_data \\
        --sr_epochs 20 --ft_epochs 30

    # 只做分析 (不訓練):
    python network_slimming_pruning.py \\
        --model_path model.pt \\
        --analyze_only

    # 跳過 sparsity training，直接從已訓練的 checkpoint 剪枝:
    python network_slimming_pruning.py \\
        --model_path model.pt \\
        --skip_sparsity_training \\
        --sparsity_checkpoint pruned_output/model_sparsity_trained.pt
"""

import os
import sys
import copy
import json
import time
import logging
import random
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# Matplotlib: 使用非互動式後端 (WSL 無 display server 也能跑)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mamba_ssm import Mamba

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# CLI 參數
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Network Slimming pruning for CNN+Mamba silent speech model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 路徑
    p.add_argument('--model_path', type=str, default='models_mamba1/transduction_model/model.pt',
                   help='預訓練模型路徑')
    p.add_argument('--data_dir', type=str, default='.',
                   help='訓練資料根目錄')
    p.add_argument('--output_dir', type=str, default='pruned_output',
                   help='輸出目錄')
    p.add_argument('--normalizers_path', type=str, default=None,
                   help='normalizers.pkl 路徑 (預設: data_dir/normalizers.pkl)')

    # 模型架構
    p.add_argument('--model_size', type=int, default=512, help='hidden dimension')
    p.add_argument('--num_layers', type=int, default=4, help='Mamba layers')
    p.add_argument('--num_outs', type=int, default=80, help='mel output dim')
    p.add_argument('--num_aux_outs', type=int, default=48, help='phoneme output dim')
    p.add_argument('--dropout', type=float, default=0.2, help='dropout')

    # Network Slimming 超參數
    p.add_argument('--sparsity_lambda', type=float, default=1e-4,
                   help='BN γ L1 正則化強度')
    p.add_argument('--prune_percent', type=float, default=0.5,
                   help='剪枝比例 (0.5 = 剪掉 50%%)')
    p.add_argument('--min_channels', type=int, default=32,
                   help='每層最少保留通道數')

    # Sparsity Training
    p.add_argument('--sr_epochs', type=int, default=20,
                   help='sparsity-aware 重訓 epoch 數')
    p.add_argument('--sr_lr', type=float, default=1e-4,
                   help='sparsity training 學習率')
    p.add_argument('--sr_batch_size', type=int, default=32, help='batch size')
    p.add_argument('--sr_seq_len', type=int, default=200, help='序列長度')

    # Fine-tuning
    p.add_argument('--ft_epochs', type=int, default=30,
                   help='fine-tune epoch 數')
    p.add_argument('--ft_lr', type=float, default=1e-4,
                   help='fine-tune 學習率')

    # 資料模式
    p.add_argument('--use_real_data', action='store_true',
                   help='使用真實 EMG 資料 (需要上傳完整資料)')

    # 流程控制
    p.add_argument('--analyze_only', action='store_true',
                   help='只做 BN γ 分析，不訓練')
    p.add_argument('--skip_sparsity_training', action='store_true',
                   help='跳過 sparsity training (從 checkpoint 開始剪枝)')
    p.add_argument('--sparsity_checkpoint', type=str, default=None,
                   help='已完成 sparsity training 的 checkpoint 路徑')
    p.add_argument('--skip_finetune', action='store_true',
                   help='跳過 fine-tuning')
    p.add_argument('--skip_benchmark', action='store_true',
                   help='跳過推論速度基準測試')
    p.add_argument('--benchmark_iters', type=int, default=100,
                   help='benchmark 迭代次數')

    # 剪枝比例掃描
    p.add_argument('--sweep_ratios', action='store_true',
                   help='掃描多種剪枝比例並列印預覽')

    return p.parse_args()


# ════════════════════════════════════════════════════════════════
# 模型架構定義 (與 architecture.py 一致)
# ════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """CNN 殘差區塊"""
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)
        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value
        return F.relu(x + res)


class MambaBlock(nn.Module):
    """Mamba SSM 區塊"""
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=8,
            d_conv=4,
            expand=1,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None,
                 model_size=512, num_layers=4, dropout=0.2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=model_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(model_size)
        self.w_out = nn.Linear(model_size, num_outs)
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids):
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:]
                x_raw[:,-r:,:] = 0
        x_raw = x_raw.transpose(1,2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x = self.w_raw_in(x_raw)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.final_norm(x)
        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        return self.w_out(x)


# ════════════════════════════════════════════════════════════════
# 剪枝後的模型架構
# ════════════════════════════════════════════════════════════════

class PrunedResBlock(nn.Module):
    """Flexible-channel CNN residual block"""
    def __init__(self, num_ins, num_mids, num_outs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_mids, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_mids)
        self.conv2 = nn.Conv1d(num_mids, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)
        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value
        return F.relu(x + res)


class PrunedModel(nn.Module):
    """Pruned CNN+Mamba model"""
    def __init__(self, channel_cfg, num_outs, num_aux_outs=None,
                 num_layers=4, dropout=0.2):
        super().__init__()
        self.channel_cfg = channel_cfg
        blocks = []
        in_ch = 8
        for mid_ch, out_ch in channel_cfg:
            blocks.append(PrunedResBlock(in_ch, mid_ch, out_ch, stride=2))
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*blocks)
        d_model = channel_cfg[-1][1]
        self.d_model = d_model
        self.w_raw_in = nn.Linear(d_model, d_model)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.w_out = nn.Linear(d_model, num_outs)
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(d_model, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids):
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:]
                x_raw[:,-r:,:] = 0
        x_raw = x_raw.transpose(1,2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x = self.w_raw_in(x_raw)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.final_norm(x)
        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        return self.w_out(x)


# ════════════════════════════════════════════════════════════════
# 工具函式
# ════════════════════════════════════════════════════════════════

def collect_bn_weights(model):
    """收集模型中所有 BatchNorm1d 的 γ 權重"""
    bn_weights = []
    bn_info = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            gamma = module.weight.data.abs().clone()
            bn_weights.append(gamma)
            bn_info.append((name, module.num_features))
    return bn_weights, bn_info


def updateBN(model, sparsity_lambda):
    """
    Apply L1 regularization gradient on all BatchNorm1d gamma.
    Call after backward() and before optimizer.step().
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            if m.weight.grad is not None:
                m.weight.grad.data.add_(sparsity_lambda * torch.sign(m.weight.data))


def compute_sparsity_loss(model):
    """Compute BN gamma L1 loss (for monitoring)"""
    l1 = 0.0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            l1 += m.weight.data.abs().sum().item()
    return l1


def count_near_zero_channels(model, threshold=0.01):
    """Count near-zero BN channels"""
    total = 0
    near_zero = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            total += m.weight.data.shape[0]
            near_zero += (m.weight.data.abs() < threshold).sum().item()
    return near_zero, total


def compute_channel_masks(model, threshold, min_channels, model_size):
    """
    Compute channel mask for each BN layer.

    Returns:
        channel_masks: dict mapping BN name -> bool tensor (True = keep)
        channel_cfg: [(mid_ch, out_ch), ...] per ResBlock
    """
    channel_masks = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d) and 'conv_blocks' in name:
            gamma = module.weight.data.abs()
            mask = gamma > threshold

            # Ensure at least min_channels are retained
            if mask.sum() < min_channels:
                _, indices = torch.topk(gamma, min_channels)
                mask = torch.zeros_like(gamma, dtype=torch.bool)
                mask[indices] = True

            channel_masks[name] = mask

    # Build channel_cfg
    channel_cfg = []
    for i in range(3):  # 3 ResBlocks
        bn1_name = f'conv_blocks.{i}.bn1'
        bn2_name = f'conv_blocks.{i}.bn2'

        mid_ch = channel_masks[bn1_name].sum().item() if bn1_name in channel_masks else model_size
        out_ch = channel_masks[bn2_name].sum().item() if bn2_name in channel_masks else model_size

        channel_cfg.append((int(mid_ch), int(out_ch)))

    return channel_masks, channel_cfg


# ════════════════════════════════════════════════════════════════
# 合成資料 (用於驗證流程)
# ════════════════════════════════════════════════════════════════

class SyntheticEMGDataset(torch.utils.data.Dataset):
    """Synthetic EMG dataset (for flow validation)"""
    def __init__(self, num_samples=500, seq_len=200, num_outs=80, num_aux_outs=48):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_speech_features = num_outs
        self.num_features = 8
        self.num_aux_outs = num_aux_outs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        raw_emg = torch.randn(self.seq_len * 8, 8) * 0.1
        emg = torch.randn(self.seq_len, 104)
        mel = torch.randn(self.seq_len, self.num_speech_features) * 0.5
        phonemes = torch.randint(0, self.num_aux_outs, (self.seq_len,))
        session_ids = torch.zeros(self.seq_len, dtype=torch.long)
        return {
            'raw_emg': raw_emg,
            'emg': emg,
            'audio_features': mel,
            'phonemes': phonemes,
            'session_ids': session_ids,
            'silent': False,
        }


def collate_synthetic(batch):
    """Collate function for synthetic data"""
    raw_emg = torch.stack([b['raw_emg'] for b in batch])
    emg = torch.stack([b['emg'] for b in batch])
    mel = torch.stack([b['audio_features'] for b in batch])
    phonemes = torch.stack([b['phonemes'] for b in batch])
    session_ids = torch.stack([b['session_ids'] for b in batch])
    return {
        'raw_emg': raw_emg,
        'emg': emg,
        'audio_features': mel,
        'phonemes': phonemes,
        'session_ids': session_ids,
        'lengths': [b['emg'].shape[0] for b in batch],
        'silent': [b['silent'] for b in batch],
    }


# ════════════════════════════════════════════════════════════════
# Loss function
# ════════════════════════════════════════════════════════════════

def simple_loss_fn(model, batch, device, use_real_data, seq_len):
    """
    Simplified loss function:
    - Synthetic data: MSE + CE
    - Real data: uses original DTW loss
    """
    if use_real_data:
        from data_utils import combine_fixed_length
        X = combine_fixed_length(
            [t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
        X_raw = combine_fixed_length(
            [t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
        sess = combine_fixed_length(
            [t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)
        pred, phoneme_pred = model(X, X_raw, sess)

        from transduction_model import dtw_loss
        loss, _ = dtw_loss(pred, phoneme_pred, batch)
        return loss
    else:
        raw_emg = batch['raw_emg'].to(device)
        emg = batch['emg'].to(device)
        mel_target = batch['audio_features'].to(device)
        phonemes_target = batch['phonemes'].to(device)
        session_ids = batch['session_ids'].to(device)

        mel_pred, phoneme_pred = model(emg, raw_emg, session_ids)

        mel_loss = F.mse_loss(mel_pred, mel_target)

        B, T, C = phoneme_pred.shape
        phoneme_loss = F.cross_entropy(
            phoneme_pred.reshape(B * T, C),
            phonemes_target.reshape(B * T)
        )

        return mel_loss + 0.5 * phoneme_loss


# ════════════════════════════════════════════════════════════════
# Weight transfer
# ════════════════════════════════════════════════════════════════

def transfer_pruned_weights(old_model, new_model, channel_masks, channel_cfg,
                            model_size, num_layers):
    """Transfer weights from sparsity-trained model to pruned model."""
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()

    d_model_old = model_size
    d_model_new = channel_cfg[-1][1]

    for i, (mid_ch, out_ch) in enumerate(channel_cfg):
        prefix = f'conv_blocks.{i}.'
        bn1_mask = channel_masks[f'conv_blocks.{i}.bn1']
        bn2_mask = channel_masks[f'conv_blocks.{i}.bn2']

        # Determine input mask
        if i == 0:
            in_mask = torch.ones(8, dtype=torch.bool)
        else:
            in_mask = channel_masks[f'conv_blocks.{i-1}.bn2']

        # conv1: [out, in, k]
        old_w = old_sd[prefix + 'conv1.weight']
        new_sd[prefix + 'conv1.weight'] = old_w[bn1_mask][:, in_mask]
        if prefix + 'conv1.bias' in old_sd:
            new_sd[prefix + 'conv1.bias'] = old_sd[prefix + 'conv1.bias'][bn1_mask]

        # bn1
        for param in ['weight', 'bias', 'running_mean', 'running_var']:
            key = prefix + 'bn1.' + param
            new_sd[key] = old_sd[key][bn1_mask]

        # conv2: [out, in, k]
        old_w = old_sd[prefix + 'conv2.weight']
        new_sd[prefix + 'conv2.weight'] = old_w[bn2_mask][:, bn1_mask]
        if prefix + 'conv2.bias' in old_sd:
            new_sd[prefix + 'conv2.bias'] = old_sd[prefix + 'conv2.bias'][bn2_mask]

        # bn2
        for param in ['weight', 'bias', 'running_mean', 'running_var']:
            key = prefix + 'bn2.' + param
            new_sd[key] = old_sd[key][bn2_mask]

        # residual_path
        if prefix + 'residual_path.weight' in old_sd:
            old_w = old_sd[prefix + 'residual_path.weight']
            new_sd[prefix + 'residual_path.weight'] = old_w[bn2_mask][:, in_mask]
            if prefix + 'residual_path.bias' in old_sd:
                new_sd[prefix + 'residual_path.bias'] = old_sd[prefix + 'residual_path.bias'][bn2_mask]

            for param in ['weight', 'bias', 'running_mean', 'running_var']:
                key = prefix + 'res_norm.' + param
                new_sd[key] = old_sd[key][bn2_mask]

    # ── Mamba layers and Linear layers ──
    last_bn2_mask = channel_masks['conv_blocks.2.bn2']

    if d_model_new == d_model_old:
        # d_model unchanged: copy directly
        for key in old_sd:
            if key.startswith('mamba_layers.') or key.startswith('final_norm.'):
                new_sd[key] = old_sd[key]

        for key in ['w_raw_in.weight', 'w_raw_in.bias',
                     'w_out.weight', 'w_out.bias']:
            if key in old_sd:
                new_sd[key] = old_sd[key]
        if 'w_aux.weight' in old_sd:
            new_sd['w_aux.weight'] = old_sd['w_aux.weight']
            new_sd['w_aux.bias'] = old_sd['w_aux.bias']
    else:
        # d_model changed: slice weights
        old_w = old_sd['w_raw_in.weight']
        new_sd['w_raw_in.weight'] = old_w[last_bn2_mask][:, last_bn2_mask]
        new_sd['w_raw_in.bias'] = old_sd['w_raw_in.bias'][last_bn2_mask]

        for layer_idx in range(num_layers):
            lp = f'mamba_layers.{layer_idx}.'

            # LayerNorm
            new_sd[lp + 'norm.weight'] = old_sd[lp + 'norm.weight'][last_bn2_mask]
            new_sd[lp + 'norm.bias'] = old_sd[lp + 'norm.bias'][last_bn2_mask]

            # in_proj: (2*d_inner, d_model)
            if lp + 'mamba.in_proj.weight' in old_sd:
                old_in_proj = old_sd[lp + 'mamba.in_proj.weight']
                mask_2x = last_bn2_mask.repeat(2)
                new_sd[lp + 'mamba.in_proj.weight'] = old_in_proj[mask_2x][:, last_bn2_mask]

            # conv1d: (d_inner, 1, d_conv)
            if lp + 'mamba.conv1d.weight' in old_sd:
                new_sd[lp + 'mamba.conv1d.weight'] = old_sd[lp + 'mamba.conv1d.weight'][last_bn2_mask]
                new_sd[lp + 'mamba.conv1d.bias'] = old_sd[lp + 'mamba.conv1d.bias'][last_bn2_mask]

            # out_proj: (d_model, d_inner)
            if lp + 'mamba.out_proj.weight' in old_sd:
                new_sd[lp + 'mamba.out_proj.weight'] = old_sd[lp + 'mamba.out_proj.weight'][last_bn2_mask][:, last_bn2_mask]
                if lp + 'mamba.out_proj.bias' in old_sd:
                    new_sd[lp + 'mamba.out_proj.bias'] = old_sd[lp + 'mamba.out_proj.bias'][last_bn2_mask]

            # x_proj: (dt_rank + 2*d_state, d_inner)
            if lp + 'mamba.x_proj.weight' in old_sd:
                new_sd[lp + 'mamba.x_proj.weight'] = old_sd[lp + 'mamba.x_proj.weight'][:, last_bn2_mask]

            # dt_proj: (d_inner, dt_rank)
            if lp + 'mamba.dt_proj.weight' in old_sd:
                new_sd[lp + 'mamba.dt_proj.weight'] = old_sd[lp + 'mamba.dt_proj.weight'][last_bn2_mask]
                new_sd[lp + 'mamba.dt_proj.bias'] = old_sd[lp + 'mamba.dt_proj.bias'][last_bn2_mask]

            # A_log, D
            if lp + 'mamba.A_log' in old_sd:
                new_sd[lp + 'mamba.A_log'] = old_sd[lp + 'mamba.A_log'][last_bn2_mask]
            if lp + 'mamba.D' in old_sd:
                new_sd[lp + 'mamba.D'] = old_sd[lp + 'mamba.D'][last_bn2_mask]

        # final_norm
        new_sd['final_norm.weight'] = old_sd['final_norm.weight'][last_bn2_mask]
        new_sd['final_norm.bias'] = old_sd['final_norm.bias'][last_bn2_mask]

        # w_out: (80, d_model)
        new_sd['w_out.weight'] = old_sd['w_out.weight'][:, last_bn2_mask]
        new_sd['w_out.bias'] = old_sd['w_out.bias']

        # w_aux: (48, d_model)
        if 'w_aux.weight' in old_sd:
            new_sd['w_aux.weight'] = old_sd['w_aux.weight'][:, last_bn2_mask]
            new_sd['w_aux.bias'] = old_sd['w_aux.bias']

    new_model.load_state_dict(new_sd, strict=True)
    return new_model


# ════════════════════════════════════════════════════════════════
# Benchmark
# ════════════════════════════════════════════════════════════════

def benchmark_model(model, name, device, seq_len, n_iters=100):
    """Measure model inference speed"""
    model.eval()
    raw = torch.randn(1, seq_len * 8, 8).to(device)
    feat = torch.randn(1, seq_len, 104).to(device)
    sess = torch.zeros(1, seq_len, dtype=torch.long).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(feat, raw, sess)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            model(feat, raw, sess)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_ms = elapsed / n_iters * 1000
    logger.info(f"  {name}: {avg_ms:.2f} ms/iter ({n_iters} iters)")
    return avg_ms


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.normalizers_path is None:
        args.normalizers_path = os.path.join(args.data_dir, 'normalizers.pkl')

    # ── Step 1: 載入模型 & 分析 BN γ ────────────────────────
    logger.info("=" * 60)
    logger.info("📊 Step 1: Loading model & analyzing BN gamma distribution")
    logger.info("=" * 60)

    model = Model(
        num_features=8,
        num_outs=args.num_outs,
        num_aux_outs=args.num_aux_outs,
        model_size=args.model_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model loaded: {total_params:,} params ({total_params/1e6:.2f}M)")

    # 分析 BN γ
    bn_weights, bn_info = collect_bn_weights(model)
    all_bn_weights = torch.cat(bn_weights)

    logger.info(f"\n{'名稱':<45} {'通道數':>6}  {'γ mean':>8}  {'γ std':>8}  {'γ min':>8}  {'γ max':>8}")
    logger.info("-" * 95)
    for (name, n_ch), gamma in zip(bn_info, bn_weights):
        logger.info(f"{name:<45} {n_ch:>6}  {gamma.mean():>8.4f}  {gamma.std():>8.4f}  "
                     f"{gamma.min():>8.4f}  {gamma.max():>8.4f}")
    logger.info(f"\n📌 Total BN channels: {all_bn_weights.shape[0]}")
    logger.info(f"   γ global stats: mean={all_bn_weights.mean():.4f}, std={all_bn_weights.std():.4f}")
    logger.info(f"   γ global range: [{all_bn_weights.min():.6f}, {all_bn_weights.max():.4f}]")

    # 視覺化 (儲存到 output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(all_bn_weights.cpu().numpy(), bins=100, color='steelblue', alpha=0.8,
            edgecolor='black', linewidth=0.3)
    ax.set_title('All BN gamma distribution', fontsize=13)
    ax.set_xlabel('|gamma|')
    ax.set_ylabel('Count')
    ax.axvline(x=all_bn_weights.mean().item(), color='red', linestyle='--',
               label=f'mean={all_bn_weights.mean():.4f}')
    ax.legend()

    ax = axes[0, 1]
    for (name, _), gamma in zip(bn_info, bn_weights):
        short_name = name.split('.')[-1] + ' (' + '.'.join(name.split('.')[:-1]) + ')'
        ax.hist(gamma.cpu().numpy(), bins=50, alpha=0.5, label=short_name)
    ax.set_title('Per-layer BN gamma distribution', fontsize=13)
    ax.set_xlabel('|gamma|')
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    sorted_bn, _ = torch.sort(all_bn_weights)
    ax.plot(sorted_bn.cpu().numpy(), linewidth=1.5, color='steelblue')
    threshold_idx = int(len(sorted_bn) * args.prune_percent)
    threshold_val = sorted_bn[threshold_idx].item()
    ax.axhline(y=threshold_val, color='red', linestyle='--',
               label=f'{args.prune_percent*100:.0f}% threshold = {threshold_val:.4f}')
    ax.axvline(x=threshold_idx, color='red', linestyle=':', alpha=0.5)
    ax.set_title('Sorted BN gamma (pruning threshold)', fontsize=13)
    ax.set_xlabel('Channel Index (sorted)')
    ax.set_ylabel('|gamma|')
    ax.legend()

    ax = axes[1, 1]
    block_names = [name for name, _ in bn_info]
    block_means = [gamma.mean().item() for gamma in bn_weights]
    block_stds = [gamma.std().item() for gamma in bn_weights]
    x_pos = range(len(block_names))
    ax.bar(x_pos, block_means, yerr=block_stds, color='steelblue', alpha=0.8, capsize=5)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([n.replace('conv_blocks.', 'B') for n in block_names],
                       rotation=45, ha='right', fontsize=9)
    ax.set_title('Mean +/- std of BN gamma per layer', fontsize=13)
    ax.set_ylabel('|gamma|')

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'bn_gamma_distribution_before_sparsity.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"📊 Plot saved: {plot_path}")

    if args.analyze_only:
        logger.info("\n🏁 --analyze_only 模式，跳過訓練步驟。")
        return

    # ── Step 2: Sparsity-aware Training ─────────────────────
    if args.skip_sparsity_training:
        logger.info("=" * 60)
        logger.info("⏭️  Skipping sparsity training (loading from checkpoint)")
        logger.info("=" * 60)

        sr_ckpt = args.sparsity_checkpoint or os.path.join(args.output_dir, 'model_sparsity_trained.pt')
        model_sr = Model(
            num_features=8,
            num_outs=args.num_outs,
            num_aux_outs=args.num_aux_outs,
            model_size=args.model_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        model_sr.load_state_dict(torch.load(sr_ckpt, map_location=device, weights_only=False), strict=True)
        model_sr.eval()
        logger.info(f"✅ Loaded sparsity-trained checkpoint: {sr_ckpt}")
    else:
        logger.info("=" * 60)
        logger.info("🎯 Step 2: Sparsity-aware Training")
        logger.info("=" * 60)
        logger.info(f"   Epochs: {args.sr_epochs}")
        logger.info(f"   Sparsity lambda: {args.sparsity_lambda}")
        logger.info(f"   Learning Rate: {args.sr_lr}")

        model_sr = Model(
            num_features=8,
            num_outs=args.num_outs,
            num_aux_outs=args.num_aux_outs,
            model_size=args.model_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        model_sr.load_state_dict(
            torch.load(args.model_path, map_location=device, weights_only=False), strict=True)
        model_sr.train()

        optimizer = torch.optim.AdamW(model_sr.parameters(), lr=args.sr_lr, weight_decay=1e-7)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sr_epochs)

        # Data loader
        if args.use_real_data:
            from absl import flags as absl_flags
            FLAGS = absl_flags.FLAGS

            # 1. 先 import，讓 data_utils 自動註冊自己模組內的 flags
            from read_emg import EMGDataset, SizeAwareSampler
            from data_utils import combine_fixed_length

            # 2. 檢查 flag 是否已存在，不存在才定義，避免 DuplicateFlagError
            def define_if_absent(define_fn, name, default_val, desc=''):
                if name not in FLAGS:
                    define_fn(name, default_val, desc)

            define_if_absent(absl_flags.DEFINE_integer, 'model_size', args.model_size)
            define_if_absent(absl_flags.DEFINE_integer, 'num_layers', args.num_layers)
            define_if_absent(absl_flags.DEFINE_float, 'dropout', args.dropout)
            define_if_absent(absl_flags.DEFINE_integer, 'batch_size', args.sr_batch_size)
            define_if_absent(absl_flags.DEFINE_float, 'phoneme_loss_weight', 0.5)
            define_if_absent(absl_flags.DEFINE_string, 'normalizers_file', args.normalizers_path)
            define_if_absent(absl_flags.DEFINE_list, 'remove_channels', [])
            define_if_absent(absl_flags.DEFINE_list, 'silent_data_directories',
                             [os.path.join(args.data_dir, 'emg_data/silent_parallel_data')])
            define_if_absent(absl_flags.DEFINE_list, 'voiced_data_directories',
                             [os.path.join(args.data_dir, 'emg_data/voiced_parallel_data'),
                              os.path.join(args.data_dir, 'emg_data/nonparallel_data')])
            define_if_absent(absl_flags.DEFINE_string, 'testset_file',
                             os.path.join(args.data_dir, 'testset_largedev.json'))
            define_if_absent(absl_flags.DEFINE_string, 'text_align_directory',
                             os.path.join(args.data_dir, 'text_alignments'))

            # 3. 確保解析
            if not FLAGS.is_parsed():
                FLAGS(sys.argv[:1])

            # 4. 將 CLI 參數的值設定給 FLAGS
            FLAGS.normalizers_file = args.normalizers_path
            FLAGS.batch_size = args.sr_batch_size
            FLAGS.silent_data_directories = [os.path.join(args.data_dir, 'emg_data/silent_parallel_data')]
            FLAGS.voiced_data_directories = [os.path.join(args.data_dir, 'emg_data/voiced_parallel_data'),
                                             os.path.join(args.data_dir, 'emg_data/nonparallel_data')]
            FLAGS.testset_file = os.path.join(args.data_dir, 'testset_largedev.json')
            FLAGS.text_align_directory = os.path.join(args.data_dir, 'text_alignments')

            trainset = EMGDataset(dev=False, test=False)
            train_loader = torch.utils.data.DataLoader(
                trainset, pin_memory=True, collate_fn=trainset.collate_raw,
                num_workers=0, batch_sampler=SizeAwareSampler(trainset, 256000)
            )
            logger.info(f"✅ Real dataset loaded: train={len(trainset)}")
        else:
            logger.info("⚠️  Using synthetic data (flow validation only)")
            synth_train = SyntheticEMGDataset(
                num_samples=500, seq_len=args.sr_seq_len,
                num_outs=args.num_outs, num_aux_outs=args.num_aux_outs)
            train_loader = torch.utils.data.DataLoader(
                synth_train, batch_size=args.sr_batch_size, shuffle=True,
                collate_fn=collate_synthetic, num_workers=0
            )

        # Training loop
        history = {
            'train_loss': [],
            'sparsity_loss': [],
            'near_zero_channels': [],
            'near_zero_ratio': [],
        }

        for epoch in range(args.sr_epochs):
            model_sr.train()
            epoch_losses = []

            pbar = tqdm.tqdm(train_loader, desc=f'SR Epoch {epoch+1}/{args.sr_epochs}')
            for batch in pbar:
                optimizer.zero_grad()

                loss = simple_loss_fn(model_sr, batch, device,
                                      args.use_real_data, args.sr_seq_len)
                loss.backward()

                # 🔑 Network Slimming core
                updateBN(model_sr, args.sparsity_lambda)

                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            scheduler.step()

            avg_loss = np.mean(epoch_losses)
            sp_loss = compute_sparsity_loss(model_sr)
            nz, total = count_near_zero_channels(model_sr, threshold=0.01)

            history['train_loss'].append(avg_loss)
            history['sparsity_loss'].append(sp_loss)
            history['near_zero_channels'].append(nz)
            history['near_zero_ratio'].append(nz / total)

            logger.info(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
                         f"BN L1={sp_loss:.2f}, "
                         f"near-zero={nz}/{total} ({nz/total*100:.1f}%)")

        # Save sparsity-trained model
        sr_model_path = os.path.join(args.output_dir, 'model_sparsity_trained.pt')
        torch.save(model_sr.state_dict(), sr_model_path)
        logger.info(f"\n✅ Sparsity-trained model saved: {sr_model_path}")

        # 視覺化 sparsity training
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(history['train_loss'], color='steelblue', linewidth=2)
        ax.set_title('Training Loss', fontsize=13)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(history['sparsity_loss'], color='orange', linewidth=2)
        ax.set_title('BN gamma L1 Sum (lower = sparser)', fontsize=13)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Sum |gamma|'); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot([r * 100 for r in history['near_zero_ratio']], color='crimson', linewidth=2)
        ax.set_title('Near-zero channel ratio (|gamma| < 0.01)', fontsize=13)
        ax.set_xlabel('Epoch'); ax.set_ylabel('%'); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        bn_weights_after, _ = collect_bn_weights(model_sr)
        all_bn_after = torch.cat(bn_weights_after)
        ax.hist(all_bn_after.cpu().numpy(), bins=100, color='steelblue', alpha=0.8,
                edgecolor='black', linewidth=0.3)
        ax.set_title('BN gamma distribution after sparsity training', fontsize=13)
        ax.set_xlabel('|gamma|'); ax.set_ylabel('Count')
        ax.axvline(x=0.01, color='red', linestyle='--', label='threshold=0.01')
        ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, 'sparsity_training_progress.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"📊 Plot saved: {plot_path}")

    # ── Step 3: Channel Pruning ─────────────────────────────
    logger.info("=" * 60)
    logger.info("✂️  Step 3: Channel Pruning")
    logger.info("=" * 60)

    # Compute global threshold
    prunable_bn_weights = []
    all_prunable_bn_weights = []

    for name, module in model_sr.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            gamma = module.weight.data.abs().clone()
            if 'bn1' in name and 'conv_blocks' in name:
                prunable_bn_weights.append(gamma)
            if ('bn1' in name or 'bn2' in name) and 'conv_blocks' in name:
                all_prunable_bn_weights.append(gamma)

    all_prunable = torch.cat(all_prunable_bn_weights)
    sorted_prunable, _ = torch.sort(all_prunable)

    threshold_idx = int(len(sorted_prunable) * args.prune_percent)
    threshold = sorted_prunable[threshold_idx].item()

    logger.info(f"📐 Pruning threshold:")
    logger.info(f"   Total prunable channels: {len(sorted_prunable)}")
    logger.info(f"   Prune ratio: {args.prune_percent*100:.0f}%")
    logger.info(f"   gamma threshold: {threshold:.6f}")

    # Determine channels to keep
    channel_masks, channel_cfg = compute_channel_masks(
        model_sr, threshold, args.min_channels, args.model_size)

    logger.info(f"\n{'ResBlock':<12} {'bn1 (mid)':>20} {'bn2 (out)':>20}")
    logger.info("-" * 55)
    for i, (mid, out) in enumerate(channel_cfg):
        logger.info(f"Block {i}      {mid:>6}/{args.model_size} "
                     f"({mid/args.model_size*100:5.1f}%)   "
                     f"{out:>6}/{args.model_size} ({out/args.model_size*100:5.1f}%)")

    new_d_model = channel_cfg[-1][1]
    logger.info(f"\n📌 New d_model (Mamba): {new_d_model} (original: {args.model_size})")

    if new_d_model != args.model_size:
        logger.info("⚠️  d_model changed! Mamba layers need rebuilding.")
    else:
        logger.info("✅ d_model unchanged, Mamba layers can be copied directly.")

    # Build pruned model
    pruned_model = PrunedModel(
        channel_cfg=channel_cfg,
        num_outs=args.num_outs,
        num_aux_outs=args.num_aux_outs,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    pruned_model = transfer_pruned_weights(
        model_sr, pruned_model, channel_masks, channel_cfg,
        args.model_size, args.num_layers)

    old_params = sum(p.numel() for p in model_sr.parameters())
    new_params = sum(p.numel() for p in pruned_model.parameters())
    logger.info(f"\n✅ Weight transfer complete!")
    logger.info(f"   Original params: {old_params:>12,} ({old_params/1e6:.2f}M)")
    logger.info(f"   Pruned params:   {new_params:>12,} ({new_params/1e6:.2f}M)")
    logger.info(f"   Compression:     {new_params/old_params*100:.1f}% "
                 f"(reduced {(1 - new_params/old_params)*100:.1f}%)")

    # Verify
    pruned_model.eval()
    model_sr.eval()
    with torch.no_grad():
        test_raw = torch.randn(1, args.sr_seq_len * 8, 8).to(device)
        test_feat = torch.randn(1, args.sr_seq_len, 104).to(device)
        test_sess = torch.zeros(1, args.sr_seq_len, dtype=torch.long).to(device)

        old_mel, old_phon = model_sr(test_feat, test_raw, test_sess)
        new_mel, new_phon = pruned_model(test_feat, test_raw, test_sess)

        logger.info(f"\nOriginal output: mel={old_mel.shape}, phoneme={old_phon.shape}")
        logger.info(f"Pruned output:   mel={new_mel.shape}, phoneme={new_phon.shape}")

        if channel_cfg[-1][1] == args.model_size:
            diff_mel = (old_mel - new_mel).abs().mean().item()
            diff_phon = (old_phon - new_phon).abs().mean().item()
            logger.info(f"Mel diff: {diff_mel:.6f}, Phoneme diff: {diff_phon:.6f}")
            if diff_mel < 0.01 and diff_phon < 0.01:
                logger.info("✅ Outputs nearly identical — weight transfer correct!")
            else:
                logger.info("⚠️  Output difference detected — fine-tuning needed")
        else:
            logger.info("📌 d_model changed, different output shapes expected. Fine-tuning required.")

    # ── Step 4: Fine-tune ───────────────────────────────────
    if args.skip_finetune:
        logger.info("\n⏭️  Skipping fine-tuning (--skip_finetune)")
    else:
        logger.info("=" * 60)
        logger.info("🔧 Step 4: Fine-tuning the Pruned Model")
        logger.info("=" * 60)
        logger.info(f"   Epochs: {args.ft_epochs}")
        logger.info(f"   Learning Rate: {args.ft_lr}")

        pruned_model.train()
        ft_optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=args.ft_lr, weight_decay=1e-7)
        ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=args.ft_epochs)

        if args.use_real_data:
            from read_emg import EMGDataset, SizeAwareSampler
            if 'trainset' not in dir():
                trainset = EMGDataset(dev=False, test=False)
            ft_loader = torch.utils.data.DataLoader(
                trainset, pin_memory=True, collate_fn=trainset.collate_raw,
                num_workers=0, batch_sampler=SizeAwareSampler(trainset, 256000)
            )
        else:
            synth_train_ft = SyntheticEMGDataset(
                num_samples=500, seq_len=args.sr_seq_len,
                num_outs=args.num_outs, num_aux_outs=args.num_aux_outs)
            ft_loader = torch.utils.data.DataLoader(
                synth_train_ft, batch_size=args.sr_batch_size, shuffle=True,
                collate_fn=collate_synthetic, num_workers=0
            )

        ft_history = {'train_loss': []}

        for epoch in range(args.ft_epochs):
            pruned_model.train()
            epoch_losses = []

            pbar = tqdm.tqdm(ft_loader, desc=f'FT Epoch {epoch+1}/{args.ft_epochs}')
            for batch in pbar:
                ft_optimizer.zero_grad()
                loss = simple_loss_fn(pruned_model, batch, device,
                                      args.use_real_data, args.sr_seq_len)
                loss.backward()
                ft_optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            ft_scheduler.step()
            avg_loss = np.mean(epoch_losses)
            ft_history['train_loss'].append(avg_loss)
            logger.info(f"  FT Epoch {epoch+1}: loss={avg_loss:.4f}")

        logger.info("✅ Fine-tuning complete")

        # Fine-tune loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(ft_history['train_loss'], color='steelblue', linewidth=2,
                 marker='o', markersize=4)
        plt.title('Fine-tuning Loss', fontsize=14)
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        ft_plot_path = os.path.join(args.output_dir, 'finetune_loss.png')
        plt.savefig(ft_plot_path, dpi=150)
        plt.close()
        logger.info(f"📊 Plot saved: {ft_plot_path}")

    # ── Step 5: Export ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("💾 Step 5: Exporting Pruned Model")
    logger.info("=" * 60)

    pruned_model_path = os.path.join(args.output_dir, 'model_pruned.pt')
    torch.save(pruned_model.state_dict(), pruned_model_path)

    channel_cfg_path = os.path.join(args.output_dir, 'channel_cfg.json')
    with open(channel_cfg_path, 'w') as f:
        json.dump(channel_cfg, f, indent=2)

    report = {
        'original_model_size': args.model_size,
        'original_params': old_params,
        'pruned_params': new_params,
        'compression_ratio': new_params / old_params,
        'prune_percent': args.prune_percent,
        'sparsity_lambda': args.sparsity_lambda,
        'threshold': threshold,
        'channel_cfg': channel_cfg,
        'new_d_model': new_d_model,
        'sr_epochs': args.sr_epochs,
        'ft_epochs': args.ft_epochs,
        'channel_details': {},
    }

    for name, mask in channel_masks.items():
        report['channel_details'][name] = {
            'original': int(mask.shape[0]),
            'retained': int(mask.sum().item()),
            'pruned': int((~mask).sum().item()),
        }

    report_path = os.path.join(args.output_dir, 'pruning_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"   Model weights:  {pruned_model_path}")
    logger.info(f"   Channel config: {channel_cfg_path}")
    logger.info(f"   Pruning report: {report_path}")

    old_size_mb = (os.path.getsize(args.model_path) / 1e6
                   if os.path.exists(args.model_path) else old_params * 4 / 1e6)
    new_size_mb = os.path.getsize(pruned_model_path) / 1e6
    logger.info(f"\n📊 Model size: {old_size_mb:.1f} MB → {new_size_mb:.1f} MB "
                 f"(reduced {(1 - new_size_mb/old_size_mb)*100:.1f}%)")

    # ── Benchmark ───────────────────────────────────────────
    if not args.skip_benchmark:
        logger.info("=" * 60)
        logger.info("⏱️  Inference Speed Benchmark")
        logger.info("=" * 60)

        original_ms = benchmark_model(model, "Original", device,
                                       args.sr_seq_len, args.benchmark_iters)
        pruned_ms = benchmark_model(pruned_model, "Pruned", device,
                                     args.sr_seq_len, args.benchmark_iters)
        logger.info(f"  Speedup: {original_ms/pruned_ms:.2f}x")

    # ── Sweep ratios ────────────────────────────────────────
    if args.sweep_ratios:
        logger.info("=" * 60)
        logger.info("📊 Pruning Ratio Sweep Preview")
        logger.info("=" * 60)
        logger.info(f"{'Prune %':>8}  {'Threshold':>10}  {'Channel Config':>40}  "
                     f"{'d_model':>8}  {'Params':>12}  {'Ratio':>8}")
        logger.info("-" * 100)

        for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            idx = int(len(sorted_prunable) * pct)
            thresh = sorted_prunable[idx].item()
            masks, cfg = compute_channel_masks(model_sr, thresh,
                                                args.min_channels, args.model_size)
            test_model = PrunedModel(
                channel_cfg=cfg, num_outs=args.num_outs,
                num_aux_outs=args.num_aux_outs,
                num_layers=args.num_layers, dropout=args.dropout
            )
            params = sum(p.numel() for p in test_model.parameters())
            d_m = cfg[-1][1]

            logger.info(f"{pct*100:>7.0f}%  {thresh:>10.6f}  {str(cfg):>40}  "
                         f"{d_m:>8}  {params:>12,}  {params/old_params*100:>7.1f}%")
            del test_model

        logger.info(f"\n📌 Original model: {old_params:,} parameters")

    # ── Final summary ───────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📋 Network Slimming Pruning — Summary")
    logger.info("=" * 60)
    logger.info(f"\n🏗️  Architecture:")
    logger.info(f"   Original: [(512,512), (512,512), (512,512)], d_model={args.model_size}")
    logger.info(f"   Pruned:   {channel_cfg}, d_model={new_d_model}")
    logger.info(f"\n📊 Parameters:")
    logger.info(f"   Original: {old_params:,} ({old_params/1e6:.2f}M)")
    logger.info(f"   Pruned:   {new_params:,} ({new_params/1e6:.2f}M)")
    logger.info(f"   Reduced:  {(1-new_params/old_params)*100:.1f}%")
    logger.info(f"\n💾 Output: {args.output_dir}/")
    logger.info("=" * 60)
    logger.info("🏁 Done!")


if __name__ == '__main__':
    main()
