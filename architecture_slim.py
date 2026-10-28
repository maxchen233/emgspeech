"""Prunable CNN-Mamba architecture for Network Slimming.

Compatible with the user's original checkpoint when cnn_mid_channels == [model_size]*3.
Only the internal channels conv1 -> bn1 -> conv2 of each residual block are pruned.
The block outputs stay at model_size, so the Mamba backend is unchanged.
"""

import random
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None


class ResBlock(nn.Module):
    def __init__(self, num_ins: int, num_outs: int, stride: int = 1,
                 mid_channels: Optional[int] = None):
        super().__init__()
        mid = int(num_outs if mid_channels is None else mid_channels)
        if mid <= 0:
            raise ValueError("mid_channels must be > 0")

        self.num_ins = int(num_ins)
        self.num_outs = int(num_outs)
        self.mid_channels = mid
        self.stride = int(stride)

        self.conv1 = nn.Conv1d(num_ins, mid, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid, num_outs, 3, padding=1)
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

    def forward_causal(self, x, cache1, cache2):
        input_value = x
        k = 3

        # conv1 cache is on block input channels
        if cache1 is not None:
            x1 = torch.cat([cache1, x], dim=-1)
        else:
            x1 = F.pad(x, (k - 1, 0))
        new_cache1 = x1[:, :, -(k - 1):].clone()
        x = F.conv1d(x1, self.conv1.weight, self.conv1.bias,
                     stride=self.conv1.stride[0])
        x = F.relu(self.bn1(x))

        # conv2 cache is on the prunable mid channels
        if cache2 is not None:
            x2 = torch.cat([cache2, x], dim=-1)
        else:
            x2 = F.pad(x, (k - 1, 0))
        new_cache2 = x2[:, :, -(k - 1):].clone()
        x = F.conv1d(x2, self.conv2.weight, self.conv2.bias,
                     stride=self.conv2.stride[0])
        x = self.bn2(x)

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res), new_cache1, new_cache2


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not importable. Install the same mamba_ssm version used "
                "to train your original CNN-Mamba model."
            )
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

    def forward_streaming(self, x, conv_state, ssm_state):
        _, T, _ = x.shape
        residual = x
        x = self.norm(x)

        outputs = []
        for t in range(T):
            xt = x[:, t:t + 1, :]
            out_t, conv_state, ssm_state = self.mamba.step(
                xt, conv_state, ssm_state
            )
            outputs.append(out_t.squeeze(1))
        x = torch.stack(outputs, dim=1)
        return residual + x, conv_state, ssm_state


class Model(nn.Module):
    """CNN-Mamba with independently prunable internal CNN channels.

    cnn_mid_channels only changes the conv1/bn1/conv2 internal width in each
    ResBlock. Each block output remains model_size, making pruning isolated
    from the Mamba backend.
    """

    def __init__(self, num_features, num_outs, num_aux_outs=None,
                 model_size: int = 512, num_layers: int = 4,
                 dropout: float = 0.2,
                 cnn_mid_channels: Optional[Iterable[int]] = None,
                 internal_random_shift: bool = True):
        super().__init__()
        self.model_size = int(model_size)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.internal_random_shift = bool(internal_random_shift)

        if cnn_mid_channels is None:
            mids = [self.model_size] * 3
        else:
            mids = [int(x) for x in cnn_mid_channels]
            if len(mids) != 3:
                raise ValueError("cnn_mid_channels must contain exactly 3 integers")
        self.cnn_mid_channels = mids

        self.conv_blocks = nn.Sequential(
            ResBlock(8, self.model_size, 2, mids[0]),
            ResBlock(self.model_size, self.model_size, 2, mids[1]),
            ResBlock(self.model_size, self.model_size, 2, mids[2]),
        )
        self.w_raw_in = nn.Linear(self.model_size, self.model_size)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=self.model_size, dropout=self.dropout_p)
            for _ in range(self.num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.model_size)
        self.w_out = nn.Linear(self.model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(self.model_size, num_aux_outs)

    @staticmethod
    def shift_raw_input(x_raw: torch.Tensor, shift: Optional[int] = None):
        """Replicate the original random [0,7] raw-EMG shift without in-place mutation."""
        if shift is None:
            shift = random.randrange(8)
        shift = int(shift)
        if shift <= 0:
            return x_raw
        y = x_raw.clone()
        y[:, :-shift, :] = x_raw[:, shift:, :]
        y[:, -shift:, :] = 0
        return y

    def forward(self, x_feat, x_raw, session_ids):
        # x_feat/session_ids are intentionally kept for checkpoint/API compatibility.
        if self.training and self.internal_random_shift:
            x_raw = self.shift_raw_input(x_raw)

        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)
        x = self.w_raw_in(x_raw)

        for layer in self.mamba_layers:
            x = layer(x)

        x = self.final_norm(x)
        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        return self.w_out(x)

    def init_streaming_state(self, device):
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
        x = x_raw.transpose(1, 2)
        for i, block in enumerate(self.conv_blocks):
            cache1, cache2 = state['cnn_caches'][i]
            x, cache1, cache2 = block.forward_causal(x, cache1, cache2)
            state['cnn_caches'][i] = (cache1, cache2)

        x = x.transpose(1, 2)
        x = self.w_raw_in(x)
        for i, layer in enumerate(self.mamba_layers):
            conv_s, ssm_s = state['mamba_states'][i]
            x, conv_s, ssm_s = layer.forward_streaming(x, conv_s, ssm_s)
            state['mamba_states'][i] = (conv_s, ssm_s)

        x = self.final_norm(x)
        mel = self.w_out(x)
        aux = self.w_aux(x) if self.has_aux_out else None
        return mel, aux, state
