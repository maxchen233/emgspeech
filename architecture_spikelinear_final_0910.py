import random

import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

from mamba_ssm import Mamba

from spike_linear_ac import SpikeLinearAC

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 512, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 4, 'number of layers')
flags.DEFINE_float('dropout', .2, 'dropout')

# ---三元脈衝與反正切替代梯度---
class SurrogateTernaryArcTan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x, threshold)
        
        pos_spike = (x >= threshold).float()
        neg_spike = (x <= -threshold).float()
        return pos_spike - neg_spike

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        alpha = 2.0  # 控制梯度平滑度
        
        grad_pos = alpha / (1.0 + (alpha * (x - threshold)) ** 2)
        grad_neg = alpha / (1.0 + (alpha * (x + threshold)) ** 2)
        
        grad_input = grad_output * (grad_pos + grad_neg)
        return grad_input, None

class TernarySpikingActivation(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.register_buffer('threshold', torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x):
        return SurrogateTernaryArcTan.apply(x, self.threshold)
# -----------------------------------

class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1, use_spike=False, spike_threshold=0.35):
        super().__init__()
        self.use_spike = use_spike
        self.spike_threshold = spike_threshold
        
        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        # 還原：移除 affine=False，恢復可學習的縮放與平移參數
        self.bn1 = nn.BatchNorm1d(num_outs) 
        
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        # 還原：移除 affine=False
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            # 還原：移除 affine=False
            self.res_norm = nn.BatchNorm1d(num_outs) 
        else:
            self.residual_path = None

        if self.use_spike:
            self.spike_act = TernarySpikingActivation(threshold=spike_threshold)

        # (已移除 self.final_bn)

    def forward(self, x):
        res = x
        if self.residual_path is not None:
            res = self.residual_path(x)
            res = self.res_norm(res)

        x = self.conv1(x)
        x = self.bn1(x)
        if self.use_spike:
            x = self.spike_act(x)
        else:
            x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # 還原：直接將 x 與 res 相加，不再經過 final_bn
        if self.use_spike:
            return self.spike_act(x + res)
        else:
            return F.relu(x + res)


class PureMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + self.dropout(x)


class PureMambaModel(nn.Module):
    def __init__(self, num_features, num_speech_features, num_phonemes):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2, use_spike=False),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2, use_spike=False),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2, use_spike=False),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)
        self.mamba_layers = nn.ModuleList([
            PureMambaBlock(d_model=FLAGS.model_size, dropout=FLAGS.dropout)
            for _ in range(FLAGS.num_layers)
        ])
        self.final_norm = nn.LayerNorm(FLAGS.model_size)
        self.w_out = nn.Linear(FLAGS.model_size, num_speech_features)
        self.w_aux = nn.Linear(FLAGS.model_size, num_phonemes)
        self.debug_compute_info = OrderedDict()

    def forward(self, _, x_raw, session_ids):
        x = x_raw.transpose(1, 2)
        x = self.conv_blocks(x).transpose(1, 2)
        self.debug_compute_info = OrderedDict()
        self.debug_compute_info['post_cnn_shape'] = tuple(x.shape)
        x = self.w_raw_in(x)
        self.debug_compute_info['post_w_raw_in_shape'] = tuple(x.shape)
        for idx, layer in enumerate(self.mamba_layers):
            x = layer(x)
            self.debug_compute_info[f'mamba_layer_{idx}_out_shape'] = tuple(x.shape)
        x = self.final_norm(x)
        self.debug_compute_info['post_final_norm_shape'] = tuple(x.shape)
        return (self.w_out(x), self.w_aux(x))

class MambaBlock(nn.Module): #for student
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + self.dropout(x)

class Model(nn.Module):
    def __init__(self, num_features, num_speech_features, n_phones, no_sgc=False, spike_threshold=1.2):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2, use_spike=True, spike_threshold=spike_threshold),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2, use_spike=True, spike_threshold=spike_threshold),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2, use_spike=True, spike_threshold=spike_threshold),
        )
        # 輸入端的初始脈衝化
        self.spike_in = TernarySpikingActivation(threshold=spike_threshold)
        
        # 5. w_raw_in 作為 Readout 層，連接脈衝與連續網路
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)
        
        # 使用純淨的 MambaBlock
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=FLAGS.model_size, dropout=FLAGS.dropout)
            for _ in range(FLAGS.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(FLAGS.model_size)
        self.w_out = nn.Linear(FLAGS.model_size, num_speech_features)
        self.has_aux_out = n_phones is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, n_phones)

    def forward(self, _, x_raw, session_ids):
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:]
                x_raw[:,-r:,:] = 0
        x_raw = x_raw.transpose(1,2)
        
        # 前端 CNN：三元脈衝域
        x_raw = self.conv_blocks(x_raw).transpose(1,2)
        # x_raw_spikes = self.spike_in(x_raw)
        # self.last_spike_tensor = x_raw_spikes 
        self.last_spike_tensor = x_raw.detach()

        
        # 橋接 Readout：將脈衝轉回連續向量
        # x = self.w_raw_in(x_raw_spikes)
        x = self.w_raw_in(x_raw)
        
        # 後端 Mamba：連續數值域
        for layer in self.mamba_layers:
            x = layer(x)
            
        x = self.final_norm(x)
        
        if self.has_aux_out:
            return (self.w_out(x), self.w_aux(x))
        else:
            return (self.w_out(x), None)

