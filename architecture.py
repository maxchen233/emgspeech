import random

import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 512, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 4, 'number of layers')
flags.DEFINE_float('dropout', .2, 'dropout')


class ResBlock(nn.Module):
    """CNN 殘差區塊 - 支援批次與串流兩種模式"""

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
        """批次模式 (訓練用) - 對稱 padding"""
        input_value = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value
        return F.relu(x + res)

    def forward_causal(self, x, cache1, cache2):
        """
        因果串流模式 - 左填充, 不看未來

        Args:
            x: (B, C, T) 新的 chunk
            cache1: (B, C_in, 2) conv1 的快取, 或 None
            cache2: (B, C_out, 2) conv2 的快取, 或 None
        Returns:
            output, new_cache1, new_cache2
        """
        input_value = x
        k = 3  # kernel_size

        # ── conv1 (causal) ────────────────────────────────────
        if cache1 is not None:
            x1 = torch.cat([cache1, x], dim=-1)
        else:
            x1 = F.pad(x, (k - 1, 0))
        new_cache1 = x1[:, :, -(k - 1):].clone()
        x = F.conv1d(x1, self.conv1.weight, self.conv1.bias,
                     stride=self.conv1.stride[0])
        x = F.relu(self.bn1(x))

        # ── conv2 (causal) ────────────────────────────────────
        if cache2 is not None:
            x2 = torch.cat([cache2, x], dim=-1)
        else:
            x2 = F.pad(x, (k - 1, 0))
        new_cache2 = x2[:, :, -(k - 1):].clone()
        x = F.conv1d(x2, self.conv2.weight, self.conv2.bias,
                     stride=self.conv2.stride[0])
        x = self.bn2(x)

        # ── residual ─────────────────────────────────────────
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res), new_cache1, new_cache2


class MambaBlock(nn.Module):
    """Mamba SSM 區塊 - 支援批次與串流兩種模式"""

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
        """批次模式 (訓練用)"""
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x

    def forward_streaming(self, x, conv_state, ssm_state):
        """
        串流模式 - 用 step() 逐時步處理, 維護 SSM 狀態

        Args:
            x: (B, T, D) 新的 chunk
            conv_state: (B, d_inner, d_conv)
            ssm_state: (B, d_inner, d_state)
        Returns:
            output: (B, T, D)
            new_conv_state, new_ssm_state
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        outputs = []
        for t in range(T):
            xt = x[:, t:t+1, :]  # (B, 1, D) - step() 要求 seqlen=1
            out_t, conv_state, ssm_state = self.mamba.step(
                xt, conv_state, ssm_state
            )
            outputs.append(out_t.squeeze(1))  # (B, D)

        x = torch.stack(outputs, dim=1)  # (B, T, D)
        # 串流時不用 dropout
        return residual + x, conv_state, ssm_state


class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=FLAGS.model_size, dropout=FLAGS.dropout)
            for _ in range(FLAGS.num_layers)
        ])

        self.final_norm = nn.LayerNorm(FLAGS.model_size)
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids):
        """批次推論 (訓練用) - 保持與 checkpoint 完全相容"""
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
        else:
            return self.w_out(x)

    def init_streaming_state(self, device):
        """初始化串流推論的所有狀態"""
        state = {
            'cnn_caches': [(None, None) for _ in range(3)],
            'mamba_states': [],
        }

        for layer in self.mamba_layers:
            d_inner = layer.mamba.d_inner
            d_conv = layer.mamba.d_conv
            d_state = layer.mamba.d_state
            conv_state = torch.zeros(1, d_inner, d_conv, device=device)
            ssm_state = torch.zeros(1, d_inner, d_state, device=device)
            state['mamba_states'].append((conv_state, ssm_state))

        return state

    def forward_streaming(self, x_raw, state):
        """
        因果串流推論 - 只處理新的 chunk, 維護狀態

        Args:
            x_raw: (1, chunk_len, 8) 新的 raw EMG chunk
            state: 由 init_streaming_state() 產生的狀態字典
        Returns:
            mel: (1, chunk_frames, 80)
            aux: (1, chunk_frames, n_aux) or None
            state: 更新後的狀態
        """
        x = x_raw.transpose(1, 2)  # (1, 8, chunk_len)

        # ── 因果 CNN 前端 ─────────────────────────────────────
        for i, block in enumerate(self.conv_blocks):
            cache1, cache2 = state['cnn_caches'][i]
            x, cache1, cache2 = block.forward_causal(x, cache1, cache2)
            state['cnn_caches'][i] = (cache1, cache2)

        x = x.transpose(1, 2)  # (1, chunk_frames, 512)
        x = self.w_raw_in(x)

        # ── Stateful Mamba ────────────────────────────────────
        for i, layer in enumerate(self.mamba_layers):
            conv_s, ssm_s = state['mamba_states'][i]
            x, conv_s, ssm_s = layer.forward_streaming(x, conv_s, ssm_s)
            state['mamba_states'][i] = (conv_s, ssm_s)

        x = self.final_norm(x)

        mel = self.w_out(x)
        aux = self.w_aux(x) if self.has_aux_out else None

        return mel, aux, state