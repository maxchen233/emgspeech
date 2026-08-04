"""
INT16 量化感知訓練 (Quantization-Aware Training, QAT)

載入已訓練的 transduction_model 權重，插入 INT16 fake-quantize 節點，
以 STE (Straight-Through Estimator) 反向傳播，讓模型適應 16-bit 低精度。

訓練完成後匯出:
  1. qat_model.pt          — QAT 微調後的 FP32 state_dict (可再做 PTQ)
  2. qat_model_int16.pt    — 真正量化為 INT16 的權重 + scale 字典

用法:
    python qat_int16.py \
        --start_training_from ./models_mamba1/transduction_model/model.pt \
        --output_directory output_qat_int16

    # 中斷後繼續:
    python qat_int16.py \
        --start_training_from ./models_mamba1/transduction_model/model.pt \
        --output_directory output_qat_int16 --resume
"""

import os
import sys
import random
import logging
import numpy as np
import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba

from read_emg import EMGDataset, SizeAwareSampler
from align import align_from_distances
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length

from absl import flags
FLAGS = flags.FLAGS

# ── Flags ──────────────────────────────────────────────────────
# 用 try/except 避免與其他腳本的 flag 衝突
_FLAG_DEFS = [
    ('integer', 'batch_size', 16, 'training batch size'),
    ('integer', 'epochs', 18, 'QAT 訓練 epochs (通常不需要太多)'),
    ('float',   'learning_rate', 1e-4, 'QAT 學習率 (比原始訓練更低)'),
    ('integer', 'learning_rate_patience', 5, 'lr decay patience'),
    ('integer', 'learning_rate_warmup', 200, 'steps of linear warmup'),
    ('float',   'l2', 1e-7, 'weight decay'),
    ('string',  'output_directory', 'output_qat_int16', 'output directory'),
    ('float',   'phoneme_loss_weight', 0.5, 'phoneme auxiliary loss weight'),
    ('string',  'start_training_from', './models_mamba1/transduction_model/model.pt',
                '要量化的原始模型權重'),
    ('integer', 'model_size', 512, '模型隱藏層大小'),
    ('integer', 'num_layers', 4, '模型層數'),
    ('float',   'dropout', 0.2, 'dropout'),
    ('float',   'data_size_fraction', 1.0, 'fraction of training data'),
    ('boolean', 'resume', False, '從上次中斷的 checkpoint 繼續訓練'),
    ('integer', 'quant_bits', 16, '量化位元數 (預設 16)'),
    ('integer', 'qat_start_epoch', 0, '從第幾個 epoch 開始啟用 fake-quant (可做 warmup)'),
    ('boolean', 'per_channel', True, '使用 per-channel 量化 (精度更高)'),
]
for _type, _name, _default, _help in _FLAG_DEFS:
    try:
        getattr(flags, f'DEFINE_{_type}')(_name, _default, _help)
    except flags.DuplicateFlagError:
        pass


# ╔══════════════════════════════════════════════════════════════╗
# ║  INT16 Fake-Quantize (STE)                                  ║
# ╚══════════════════════════════════════════════════════════════╝

class FakeQuantizeINT16(torch.autograd.Function):
    """
    模擬 INT16 量化: 前向做 round + clamp, 反向用 STE 直通梯度。
    支援 per-tensor 與 per-channel (沿 dim=0) 兩種模式。
    """
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
        x_dq = (x_q - zero_point) * scale
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        # STE: 梯度直通, 不做 clamp mask (INT16 範圍夠大, 溢出極少)
        return grad_output, None, None, None, None


def fake_quantize_int16(x, per_channel=True, bits=16):
    """對 tensor 做 INT16 fake-quantize"""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    if per_channel and x.dim() >= 2:
        # per-channel: 沿 output channel (dim=0) 計算 scale
        flat = x.view(x.size(0), -1)
        x_min = flat.min(dim=1).values
        x_max = flat.max(dim=1).values
        # 確保 scale > 0
        scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-12)
        zero_point = torch.round(qmin - x_min / scale)
        # reshape for broadcast
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        scale = scale.view(shape)
        zero_point = zero_point.view(shape)
    else:
        # per-tensor
        x_min = x.min()
        x_max = x.max()
        scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-12)
        zero_point = torch.round(qmin - x_min / scale)

    return FakeQuantizeINT16.apply(x, scale, zero_point, qmin, qmax)


# ╔══════════════════════════════════════════════════════════════╗
# ║  QAT 包裝模組                                               ║
# ╚══════════════════════════════════════════════════════════════╝

class QuantizedConv1d(nn.Module):
    """包裝 nn.Conv1d, 在 forward 時對 weight 做 fake-quant"""
    def __init__(self, conv: nn.Conv1d, per_channel=True, bits=16):
        super().__init__()
        self.conv = conv
        self.per_channel = per_channel
        self.bits = bits
        self.fake_quant_enabled = True

    def forward(self, x):
        if self.fake_quant_enabled and self.training:
            w_q = fake_quantize_int16(self.conv.weight, self.per_channel, self.bits)
        else:
            w_q = self.conv.weight
        return F.conv1d(x, w_q, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)


class QuantizedLinear(nn.Module):
    """包裝 nn.Linear, 在 forward 時對 weight 做 fake-quant"""
    def __init__(self, linear: nn.Linear, per_channel=True, bits=16):
        super().__init__()
        self.linear = linear
        self.per_channel = per_channel
        self.bits = bits
        self.fake_quant_enabled = True

    def forward(self, x):
        if self.fake_quant_enabled and self.training:
            w_q = fake_quantize_int16(self.linear.weight, self.per_channel, self.bits)
        else:
            w_q = self.linear.weight
        return F.linear(x, w_q, self.linear.bias)


class ActivationQuantHook(nn.Module):
    """插入在 activation 後, 對 activation 做 fake-quant"""
    def __init__(self, bits=16):
        super().__init__()
        self.bits = bits
        self.fake_quant_enabled = True

    def forward(self, x):
        if self.fake_quant_enabled and self.training:
            return fake_quantize_int16(x, per_channel=False, bits=self.bits)
        return x


class MambaWeightQuantHook:
    """
    用 forward hooks 對 Mamba 內部權重做 fake-quant。
    原理:
      - pre_hook:  儲存原始 weight.data, 替換為量化版本
      - post_hook: 還原原始 weight.data
    效果: forward 用量化權重 (模擬 INT16), backward 用原始 FP32 權重 (STE)。
    """
    def __init__(self, per_channel=True, bits=16):
        self.per_channel = per_channel
        self.bits = bits
        self.enabled = True
        self._saved = {}          # {param_id: original_data}
        self._param_map = {}      # {param_id: param_ref}

    def pre_hook(self, module, args):
        """forward 前: 替換權重為量化版本"""
        if not self.enabled:
            return
        self._saved.clear()
        self._param_map.clear()
        for name, param in module.named_parameters():
            if param.dim() >= 2:  # 只量化矩陣, 跳過 bias / 1-d params
                pid = id(param)
                self._saved[pid] = param.data.clone()
                self._param_map[pid] = param
                with torch.no_grad():
                    q_data = fake_quantize_int16(
                        param.data, self.per_channel, self.bits
                    )
                    param.data.copy_(q_data)

    def post_hook(self, module, args, output):
        """forward 後: 還原原始權重 (讓 backward 走 STE)"""
        for pid, orig_data in self._saved.items():
            param = self._param_map[pid]
            with torch.no_grad():
                param.data.copy_(orig_data)
        self._saved.clear()
        self._param_map.clear()
        return output


# ╔══════════════════════════════════════════════════════════════╗
# ║  模型元件 (與 architecture.py 相同結構, 加入 QAT 節點)       ║
# ╚══════════════════════════════════════════════════════════════╝

class ResBlock(nn.Module):
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
        res = self.res_norm(self.residual_path(input_value)) if self.residual_path else input_value
        return F.relu(x + res)


class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()
        ms = FLAGS.model_size

        self.conv_blocks = nn.Sequential(
            ResBlock(8, ms, 2),
            ResBlock(ms, ms, 2),
            ResBlock(ms, ms, 2),
        )
        self.w_raw_in = nn.Linear(ms, ms)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=ms, dropout=FLAGS.dropout)
            for _ in range(FLAGS.num_layers)
        ])

        self.final_norm = nn.LayerNorm(ms)
        self.w_out = nn.Linear(ms, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(ms, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids):
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]
                x_raw[:, -r:, :] = 0

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  QAT 插入工具                                               ║
# ╚══════════════════════════════════════════════════════════════╝

# 全域列表, 追蹤所有 Mamba hook 實例以便統一開關
_mamba_quant_hooks = []

def insert_fake_quant(model, per_channel=True, bits=16):
    """
    對整個模型插入 INT16 fake-quantize:
      1. Conv1d / Linear (非 Mamba 內部): 包裝為 QuantizedConv1d / QuantizedLinear
      2. Mamba 內部: 用 forward pre/post hook 在 forward 時暫時替換權重
      3. 跳過 BatchNorm / LayerNorm
      4. 在 ResBlock 後插入 activation 量化
    """
    global _mamba_quant_hooks
    _mamba_quant_hooks.clear()

    # ── 收集 Mamba 路徑, 用來區分 Mamba 內外 ──────────────────
    mamba_prefixes = set()
    for name, module in model.named_modules():
        if isinstance(module, Mamba):
            mamba_prefixes.add(name)

    def _is_inside_mamba(full_name):
        for prefix in mamba_prefixes:
            if full_name == prefix or full_name.startswith(prefix + '.'):
                return True
        return False

    # ── 1) 非 Mamba 的 Conv1d / Linear: 用包裝類 ─────────────
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            continue
        if isinstance(module, (QuantizedConv1d, QuantizedLinear)):
            continue
        if _is_inside_mamba(name):
            continue

        for attr_name, child in list(module.named_children()):
            child_full_name = f'{name}.{attr_name}' if name else attr_name
            if isinstance(child, Mamba) or _is_inside_mamba(child_full_name):
                continue
            if isinstance(child, nn.Conv1d):
                replacements.append((module, attr_name,
                                     QuantizedConv1d(child, per_channel, bits)))
            elif isinstance(child, nn.Linear):
                replacements.append((module, attr_name,
                                     QuantizedLinear(child, per_channel, bits)))

    for parent, attr_name, wrapped in replacements:
        setattr(parent, attr_name, wrapped)

    # ── 2) Mamba 模組: 用 forward hook 量化權重 ──────────────
    for name, module in model.named_modules():
        if isinstance(module, Mamba):
            hook = MambaWeightQuantHook(per_channel, bits)
            module.register_forward_pre_hook(hook.pre_hook)
            module.register_forward_hook(hook.post_hook)
            _mamba_quant_hooks.append(hook)

    # ── 3) ResBlock 後插入 activation quant ───────────────────
    new_blocks = []
    for block in model.conv_blocks:
        new_blocks.append(block)
        if isinstance(block, ResBlock):
            new_blocks.append(ActivationQuantHook(bits))
    model.conv_blocks = nn.Sequential(*new_blocks)

    n_mamba = len(_mamba_quant_hooks)
    n_wrap = len(replacements)
    logging.info(f'   量化節點: {n_wrap} 個包裝層 + {n_mamba} 個 Mamba hook')

    return model


def set_fake_quant_enabled(model, enabled: bool):
    """啟用或停用所有 fake-quant 節點 (含 Mamba hook)"""
    for m in model.modules():
        if isinstance(m, (QuantizedConv1d, QuantizedLinear, ActivationQuantHook)):
            m.fake_quant_enabled = enabled
    # 同步切換 Mamba hook
    for hook in _mamba_quant_hooks:
        hook.enabled = enabled


# ╔══════════════════════════════════════════════════════════════╗
# ║  真正量化匯出 (INT16 weight + scale)                         ║
# ╚══════════════════════════════════════════════════════════════╝

def export_int16_weights(model, bits=16):
    """
    將 QAT 完成的模型權重真正量化為 INT16, 回傳:
      - quantized_state: {name: int16_tensor}
      - scale_dict:      {name: (scale, zero_point)}
      - other_state:     非量化的參數 (norm, bias 等)
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    quantized_state = {}
    scale_dict = {}
    other_state = {}

    for name, param in model.state_dict().items():
        # 只量化 weight, 不量化 bias / norm
        is_weight = name.endswith('.weight') and param.dim() >= 2
        # 跳過 norm 層
        is_norm = any(k in name for k in ['bn1', 'bn2', 'res_norm', 'norm', 'final_norm'])

        if is_weight and not is_norm:
            flat = param.view(param.size(0), -1)
            x_min = flat.min(dim=1).values
            x_max = flat.max(dim=1).values
            scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-12)
            zero_point = torch.round(qmin - x_min / scale)

            shape = [param.size(0)] + [1] * (param.dim() - 1)
            s = scale.view(shape)
            zp = zero_point.view(shape)

            q = torch.clamp(torch.round(param / s + zp), qmin, qmax).to(torch.int16)
            quantized_state[name] = q
            scale_dict[name] = (scale.cpu(), zero_point.cpu())
        else:
            other_state[name] = param

    return quantized_state, scale_dict, other_state


# ╔══════════════════════════════════════════════════════════════╗
# ║  Loss (與 transduction_model.py 相同)                        ║
# ╚══════════════════════════════════════════════════════════════╝

def dtw_loss(predictions, phoneme_predictions, example):
    device = predictions.device
    predictions = torch.clamp(predictions, -50, 50)
    predictions = decollate_tensor(predictions, example['lengths'])
    phoneme_predictions = decollate_tensor(phoneme_predictions, example['lengths'])
    audio_features = [t.to(device, non_blocking=True) for t in example['audio_features']]
    phoneme_targets = example['phonemes']

    losses = []
    total_length = 0
    for pred, y, pred_phone, y_phone, silent in zip(
        predictions, audio_features, phoneme_predictions,
        phoneme_targets, example['silent']
    ):
        y_phone = y_phone.to(device)
        if silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
            pred_phone_log = F.log_softmax(pred_phone, -1)
            costs = dists + FLAGS.phoneme_loss_weight * -pred_phone_log[:, y_phone]
            alignment = align_from_distances(costs.T.cpu().detach().numpy())
            loss = costs[alignment, range(len(alignment))].sum()
        else:
            dists = F.pairwise_distance(y, pred)
            phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction='sum')
            loss = dists.sum() + FLAGS.phoneme_loss_weight * phoneme_loss

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses.append(loss)
        total_length += y.size(0)

    if total_length == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return sum(losses) / total_length


# ╔══════════════════════════════════════════════════════════════╗
# ║  Validation                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def validate(model, devset, device):
    model.eval()
    set_fake_quant_enabled(model, True)  # 驗證時也模擬量化
    dataloader = torch.utils.data.DataLoader(devset, batch_size=32, collate_fn=devset.collate_raw)
    losses = []
    seq_len = 200

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation', disable=None):
            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)
            pred, phoneme_pred = model(X, X_raw, sess)
            loss = dtw_loss(pred, phoneme_pred, batch)
            losses.append(loss.item())

    model.train()
    return np.mean(losses)


# ╔══════════════════════════════════════════════════════════════╗
# ║  QAT Training Loop                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def train_qat(trainset, devset, device):
    n_epochs = FLAGS.epochs
    n_phones = len(phoneme_inventory)

    # ── 建立模型 ───────────────────────────────────────────────
    model = Model(devset.num_features, devset.num_speech_features, n_phones).to(device)

    # ── 載入預訓練權重 ────────────────────────────────────────
    if FLAGS.start_training_from:
        sd = torch.load(FLAGS.start_training_from, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        logging.info(f'✅ 載入預訓練模型: {FLAGS.start_training_from}')

    s_params = sum(p.numel() for p in model.parameters())
    logging.info(f'模型參數量: {s_params:,} | FP32={s_params*4/1024/1024:.1f}MB')

    # ── 插入 fake-quant 節點 ──────────────────────────────────
    model = insert_fake_quant(model, FLAGS.per_channel, FLAGS.quant_bits)
    logging.info(f'🔧 已插入 INT{FLAGS.quant_bits} fake-quantize 節點 (per_channel={FLAGS.per_channel})')

    # ── 先做 FP32 baseline 驗證 ──────────────────────────────
    set_fake_quant_enabled(model, False)
    fp32_val = validate(model, devset, device)
    logging.info(f'📏 FP32 baseline val_loss = {fp32_val:.4f}')

    # ── 啟用 fake-quant 後的 baseline ────────────────────────
    set_fake_quant_enabled(model, True)
    q_val = validate(model, devset, device)
    logging.info(f'📏 INT{FLAGS.quant_bits} (量化前QAT) val_loss = {q_val:.4f}')
    logging.info(f'📏 量化精度落差 = {q_val - fp32_val:+.4f}')

    # ── Optimizer ──────────────────────────────────────────────
    optim = torch.optim.AdamW(model.parameters(), weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', 0.5, patience=FLAGS.learning_rate_patience
    )
    target_lr = FLAGS.learning_rate

    def set_lr(lr):
        for pg in optim.param_groups:
            pg['lr'] = lr

    def schedule_lr(it):
        if it + 1 <= FLAGS.learning_rate_warmup:
            set_lr((it + 1) * target_lr / FLAGS.learning_rate_warmup)

    # ── DataLoader ─────────────────────────────────────────────
    training_subset = trainset if FLAGS.data_size_fraction >= 1 else trainset.subset(FLAGS.data_size_fraction)
    dataloader = torch.utils.data.DataLoader(
        training_subset, pin_memory=(device == 'cuda'),
        collate_fn=devset.collate_raw, num_workers=0,
        batch_sampler=SizeAwareSampler(training_subset, 64000),
    )

    seq_len = 200
    batch_idx = 0
    best_val = float('inf')
    start_epoch = 0

    # ── Resume ─────────────────────────────────────────────────
    ckpt_path = os.path.join(FLAGS.output_directory, 'qat_checkpoint.pt')
    if FLAGS.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_scheduler'])
        start_epoch = ckpt['epoch'] + 1
        batch_idx = ckpt['batch_idx']
        best_val = ckpt['best_val']
        logging.info(f'⏩ 從 Epoch {start_epoch} 繼續 (best_val={best_val:.4f})')

    # ── Training ───────────────────────────────────────────────
    logging.info(f'\n🚀 開始 INT{FLAGS.quant_bits} QAT: {n_epochs} epochs, lr={FLAGS.learning_rate}')

    for epoch in range(start_epoch, n_epochs):
        model.train()

        # 可選: 前幾個 epoch 不啟用 fake-quant (漸進式)
        if epoch < FLAGS.qat_start_epoch:
            set_fake_quant_enabled(model, False)
            logging.info(f'Epoch {epoch+1}: fake-quant OFF (warmup)')
        else:
            set_fake_quant_enabled(model, True)

        ep_losses = []
        for batch in tqdm.tqdm(dataloader, f'Epoch {epoch+1}', disable=None):
            optim.zero_grad()
            schedule_lr(batch_idx)

            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            pred, phoneme_pred = model(X, X_raw, sess)
            loss = dtw_loss(pred, phoneme_pred, batch)

            if torch.isnan(loss) or torch.isinf(loss):
                optim.zero_grad()
                batch_idx += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            ep_losses.append(loss.item())
            batch_idx += 1

        # ── Epoch 結束 ────────────────────────────────────────
        val_loss = validate(model, devset, device)
        lr_sched.step(val_loss)

        train_loss = np.mean(ep_losses) if ep_losses else float('nan')
        logging.info(
            f'Epoch {epoch+1}/{n_epochs} | '
            f'train={train_loss:.4f} | val={val_loss:.4f}'
        )

        # 儲存最新模型
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, 'qat_model.pt'))

        # 完整 checkpoint
        torch.save({
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'lr_scheduler': lr_sched.state_dict(),
            'epoch': epoch,
            'batch_idx': batch_idx,
            'best_val': best_val,
        }, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, 'qat_best.pt'))
            logging.info(f'  💾 Best model saved (val={best_val:.4f})')

    # ── QAT 完成, 匯出 INT16 ─────────────────────────────────
    logging.info(f'\n✅ QAT 訓練完成! Best val={best_val:.4f}')

    # 載入最佳模型做匯出
    best_sd = torch.load(os.path.join(FLAGS.output_directory, 'qat_best.pt'),
                         map_location=device, weights_only=False)
    model.load_state_dict(best_sd)

    quantized_state, scale_dict, other_state = export_int16_weights(model, FLAGS.quant_bits)
    export_path = os.path.join(FLAGS.output_directory, 'qat_model_int16.pt')
    torch.save({
        'quantized_weights': quantized_state,
        'scales': scale_dict,
        'other_params': other_state,
        'bits': FLAGS.quant_bits,
        'per_channel': FLAGS.per_channel,
    }, export_path)

    # 統計量化後大小
    q_size = sum(t.nelement() * 2 for t in quantized_state.values())  # INT16 = 2 bytes
    o_size = sum(t.nelement() * 4 for t in other_state.values())      # FP32 = 4 bytes
    s_size = sum(s.nelement() * 4 + z.nelement() * 4 for s, z in scale_dict.values())
    total_mb = (q_size + o_size + s_size) / 1024 / 1024
    fp32_mb = s_params * 4 / 1024 / 1024

    logging.info(f'\n📊 量化匯出結果:')
    logging.info(f'   FP32 模型大小:  {fp32_mb:.1f} MB')
    logging.info(f'   INT16 模型大小: {total_mb:.1f} MB')
    logging.info(f'   壓縮比: {fp32_mb/total_mb:.2f}x')
    logging.info(f'   量化權重: {len(quantized_state)} 個 tensor')
    logging.info(f'   非量化參數: {len(other_state)} 個 tensor (norm/bias)')
    logging.info(f'   匯出路徑: {export_path}')

    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(FLAGS.output_directory, 'qat_log.txt'),
                'a' if FLAGS.resume else 'w'
            ),
            logging.StreamHandler(),
        ],
        level=logging.INFO, format='%(message)s',
    )

    logging.info('=' * 60)
    logging.info(f'INT{FLAGS.quant_bits} Quantization-Aware Training (QAT)')
    logging.info('=' * 60)

    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info(f'Train: {len(trainset)} | Dev: {len(devset)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Device: {device}')

    train_qat(trainset, devset, device)


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
