"""
知識蒸餾訓練腳本 — 雙 Teacher → SNN+Mamba Student

Teacher 1: Mamba (models_mamba1/transduction_model/model.pt)
           架構: architecture.Model, model_size=512, num_layers=4
Teacher 2: Transformer (transduction_model.pt)
           架構: architecture_transformer.TransformerModel, model_size=768, num_layers=6

Student:   SNN+Mamba, num_layers=4, model_size=256 (teacher1 的一半)

蒸餾策略:
  1. 輸出層蒸餾 (output KD): teacher soft target vs student output
  2. 中間層對齊 (hint loss):  teacher hidden → projector → student hidden
  3. 原始 DTW 任務 loss (ground truth)
  4. Phoneme 輔助 loss

用法:
    python distill_snn_mamba.py \
        --hifigan_checkpoint hifigan_finetuned/checkpoint \
        --output_directory output_distill

"""

import os
import sys
import math
import random
import logging
import subprocess

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from read_emg import EMGDataset, SizeAwareSampler
from align import align_from_distances
from data_utils import (
    phoneme_inventory, decollate_tensor, combine_fixed_length,
)
from transformer import TransformerEncoderLayer

from absl import flags

FLAGS = flags.FLAGS

# ── 蒸餾專用 flags ─────────────────────────────────────────────
flags.DEFINE_integer('batch_size', 16, 'training batch size')
flags.DEFINE_integer('epochs', 80, 'number of training epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('learning_rate_patience', 5, 'lr decay patience')
flags.DEFINE_integer('learning_rate_warmup', 500, 'steps of linear warmup')
flags.DEFINE_float('l2', 1e-7, 'weight decay')
flags.DEFINE_string('output_directory', 'output_distill', 'output directory')
flags.DEFINE_float('phoneme_loss_weight', 0.5, 'phoneme auxiliary loss weight')

flags.DEFINE_string('mamba_teacher_path',
                    'models_mamba1/transduction_model/model.pt',
                    'Mamba teacher checkpoint')
flags.DEFINE_string('transformer_teacher_path',
                    'transduction_model.pt',
                    'Transformer teacher checkpoint')

flags.DEFINE_float('alpha_kd', 0.5, '蒸餾 loss 權重 (vs task loss)')
flags.DEFINE_float('alpha_hint', 0.2, '中間層 hint loss 權重')
flags.DEFINE_float('temperature', 4.0, '蒸餾溫度')
flags.DEFINE_float('teacher_weight_mamba', 0.5,
                   'Mamba teacher 在蒸餾 loss 中的權重 (另一半給 Transformer)')
flags.DEFINE_integer('student_model_size', 256, 'student 隱藏層大小')
flags.DEFINE_integer('student_num_layers', 4, 'student 層數')
flags.DEFINE_float('dropout', 0.2, 'dropout')

# ── 被 read_emg / data_utils 需要但可能未定義的 flags ───────────
# 這些 flag 可能已在其他模組被定義 (DEFINE_*), 這裡用 try 包住避免重複
for _name, _default, _help in [
    ('model_size', 512, '(teacher) model hidden size — 由 architecture.py 定義'),
    ('num_layers', 4, '(teacher) num layers — 由 architecture.py 定義'),
]:
    try:
        flags.DEFINE_integer(_name, _default, _help)
    except flags.DuplicateFlagError:
        pass

try:
    flags.DEFINE_float('data_size_fraction', 1.0, 'fraction of training data')
except flags.DuplicateFlagError:
    pass


# ╔══════════════════════════════════════════════════════════════╗
# ║  SNN Neuron Models                                         ║
# ╚══════════════════════════════════════════════════════════════╝

class SurrogateSigmoid(torch.autograd.Function):
    """可微分的脈衝代理梯度 (Surrogate Gradient)"""
    sigma = 10.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sgax = SurrogateSigmoid.sigma * x
        grad = grad_output * SurrogateSigmoid.sigma / (
            (1.0 + torch.abs(sgax)) ** 2
        )
        return grad


spike_fn = SurrogateSigmoid.apply


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire 神經元

    膜電位動態: V[t] = β·V[t-1] + X[t] − V_th·spike[t-1]
    脈衝: spike[t] = Θ(V[t] − V_th)

    β (leak): 0.9 → 長時記憶, 0.5 → 快速衰減
    """

    def __init__(self, d_model, beta=0.8, threshold=1.0, learnable_beta=True):
        super().__init__()
        self.threshold = threshold
        if learnable_beta:
            # 每個通道獨立的 leak 因子, sigmoid 限制在 (0, 1)
            self.beta_logit = nn.Parameter(
                torch.full((d_model,), math.log(beta / (1 - beta)))
            )
        else:
            self.register_buffer(
                'beta_logit',
                torch.full((d_model,), math.log(beta / (1 - beta)))
            )

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) — 連續值輸入
        Returns:
            spikes: (B, T, D) — 二值脈衝 (前向 0/1, 反向有梯度)
        """
        B, T, D = x.shape
        beta = self.beta  # (D,)
        mem = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        spikes_list = []

        for t in range(T):
            mem = beta * mem + x[:, t, :]
            spike = spike_fn(mem - self.threshold)
            mem = mem - self.threshold * spike  # reset
            spikes_list.append(spike)

        return torch.stack(spikes_list, dim=1)  # (B, T, D)


# ╔══════════════════════════════════════════════════════════════╗
# ║  SNN + Mamba Student Architecture                          ║
# ╚══════════════════════════════════════════════════════════════╝

class ResBlock(nn.Module):
    """CNN 殘差區塊 — 與 teacher 相同結構"""

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


class SNNMambaBlock(nn.Module):
    """
    SNN + Mamba 混合區塊

    流程: LayerNorm → LIF Spike Encoding → Mamba SSM → Dropout → Residual

    LIF 把連續特徵轉為稀疏脈衝, Mamba 在脈衝域做序列建模,
    結合 SNN 的能效 + SSM 的長程建模能力。
    """

    def __init__(self, d_model, dropout=0.2, beta=0.8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lif = LIFNeuron(d_model, beta=beta, learnable_beta=True)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=8,
            d_conv=4,
            expand=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) — 連續特徵
        Returns:
            (B, T, D) — 殘差連接後的輸出
        """
        residual = x
        x = self.norm(x)
        x = self.lif(x)          # 連續 → 脈衝
        x = self.mamba(x)        # 脈衝域序列建模
        x = self.dropout(x)
        return residual + x


class StudentModel(nn.Module):
    """
    SNN+Mamba Student 模型

    與 teacher Mamba Model 結構對應:
      - 相同 CNN 前端 (3 × ResBlock, stride=2)
      - 相同層數 (num_layers)
      - 隱藏維度減半 (model_size // 2)
      - MambaBlock → SNNMambaBlock (加入 LIF 脈衝層)
    """

    def __init__(self, num_features, num_outs, num_aux_outs=None,
                 model_size=256, num_layers=4, dropout=0.2):
        super().__init__()
        self.model_size = model_size

        # CNN 前端 — 與 teacher 結構相同, 只是 channel 縮小
        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)

        # SNN+Mamba 層
        self.snn_mamba_layers = nn.ModuleList([
            SNNMambaBlock(d_model=model_size, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_size)
        self.w_out = nn.Linear(model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids, return_hidden=False):
        """
        Args:
            x_feat: unused (保持介面相容)
            x_raw:  (B, T*8, 8)
            session_ids: unused (保持介面相容)
            return_hidden: 是否同時回傳中間層 hidden (用於 hint loss)
        """
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]
                x_raw[:, -r:, :] = 0

        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)
        x = self.w_raw_in(x_raw)

        hiddens = []
        for layer in self.snn_mamba_layers:
            x = layer(x)
            if return_hidden:
                hiddens.append(x)

        x = self.final_norm(x)

        output = self.w_out(x)
        aux = self.w_aux(x) if self.has_aux_out else None

        if return_hidden:
            return output, aux, hiddens
        if self.has_aux_out:
            return output, aux
        return output


# ╔══════════════════════════════════════════════════════════════╗
# ║  Teacher Model Wrappers                                    ║
# ╚══════════════════════════════════════════════════════════════╝

class _MambaBlockTeacher(nn.Module):
    """Mamba 區塊 — 與 architecture.MambaBlock state_dict key 完全對應"""

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


class MambaTeacher(nn.Module):
    """載入 Mamba Teacher (architecture.py Model)"""

    def __init__(self, num_features, num_outs, num_aux_outs,
                 model_size=512, num_layers=4, dropout=0.2):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)

        self.mamba_layers = nn.ModuleList([
            _MambaBlockTeacher(d_model=model_size, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_size)
        self.w_out = nn.Linear(model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids, return_hidden=False):
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)
        x = self.w_raw_in(x_raw)

        hiddens = []
        for layer in self.mamba_layers:
            x = layer(x)
            if return_hidden:
                hiddens.append(x)

        x = self.final_norm(x)

        out = self.w_out(x)
        aux = self.w_aux(x) if self.has_aux_out else None

        if return_hidden:
            return out, aux, hiddens
        if self.has_aux_out:
            return out, aux
        return out


class TransformerTeacher(nn.Module):
    """載入 Transformer Teacher (Gaddy 原論文架構)"""

    def __init__(self, num_features, num_outs, num_aux_outs=None,
                 model_size=768, num_layers=6, nhead=8,
                 dim_feedforward=3072, dropout=0.2):
        super().__init__()
        self.model_size = model_size

        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)

        # 手動堆疊, 避免 nn.TransformerEncoder 檢查 self_attn.batch_first
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=model_size,
                nhead=nhead,
                relative_positional=True,
                relative_positional_distance=100,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.w_out = nn.Linear(model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids, return_hidden=False):
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)
        x = self.w_raw_in(x_raw)

        x = x.transpose(0, 1)   # (T, B, D) for Transformer
        for layer in self.transformer:
            x = layer(x)
        x = x.transpose(0, 1)   # (B, T, D)

        out = self.w_out(x)
        aux = self.w_aux(x) if self.has_aux_out else None

        if return_hidden:
            return out, aux, [x]
        if self.has_aux_out:
            return out, aux
        return out


# ╔══════════════════════════════════════════════════════════════╗
# ║  Hint Projector (維度對齊)                                  ║
# ╚══════════════════════════════════════════════════════════════╝

class HintProjector(nn.Module):
    """
    將 student 的 hidden (d_student) 投影到 teacher 的 hidden (d_teacher)
    用於計算中間層 hint loss
    """

    def __init__(self, d_student, d_teacher):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_student, d_teacher),
            nn.ReLU(),
            nn.Linear(d_teacher, d_teacher),
        )

    def forward(self, x):
        return self.proj(x)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Loss Functions                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def distillation_loss_mse(student_logits, teacher_logits, temperature):
    """
    基於溫度縮放的蒸餾 loss — 用 MSE 在 soft target 空間

    對於回歸任務 (mel spectrogram prediction), 不適合用 KL divergence,
    改用 temperature-scaled MSE。
    """
    # mel 輸出是連續值回歸, 直接用 MSE
    return F.mse_loss(student_logits, teacher_logits)


def distillation_loss_phoneme_kd(student_phoneme, teacher_phoneme, temperature):
    """
    Phoneme 分類的 KD loss — 用 KL divergence + temperature scaling
    """
    if student_phoneme is None or teacher_phoneme is None:
        # 回傳與計算圖相容的零 loss (正確 device)
        if student_phoneme is not None:
            return torch.tensor(0.0, device=student_phoneme.device)
        if teacher_phoneme is not None:
            return torch.tensor(0.0, device=teacher_phoneme.device)
        return torch.tensor(0.0)

    T = temperature
    student_log_prob = F.log_softmax(student_phoneme / T, dim=-1)
    teacher_prob = F.softmax(teacher_phoneme / T, dim=-1)

    # KL(teacher || student) * T^2 (Hinton 2015 的標準做法)
    kd_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (T ** 2)
    return kd_loss


def hint_loss(student_hidden, teacher_hidden, projector):
    """中間層 hint loss — 將 student hidden 投影到 teacher 空間後計算 MSE"""
    student_proj = projector(student_hidden)
    return F.mse_loss(student_proj, teacher_hidden.detach())


def dtw_loss(predictions, phoneme_predictions, example):
    """原始 DTW loss — 與 transduction_model.py 完全相同"""
    device = predictions.device
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
        assert len(pred.size()) == 2 and len(y.size()) == 2
        y_phone = y_phone.to(device)

        if silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
            costs = dists.squeeze(0)
            pred_phone_log = F.log_softmax(pred_phone, -1)
            phone_lprobs = pred_phone_log[:, y_phone]
            costs = costs + FLAGS.phoneme_loss_weight * -phone_lprobs
            alignment = align_from_distances(costs.T.cpu().detach().numpy())
            loss = costs[alignment, range(len(alignment))].sum()
        else:
            assert y.size(0) == pred.size(0)
            dists = F.pairwise_distance(y, pred)
            phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction='sum')
            loss = dists.sum() + FLAGS.phoneme_loss_weight * phoneme_loss

        losses.append(loss)
        total_length += y.size(0)

    return sum(losses) / total_length


# ╔══════════════════════════════════════════════════════════════╗
# ║  Teacher 載入                                              ║
# ╚══════════════════════════════════════════════════════════════╝

def load_mamba_teacher(num_features, num_speech_features, n_phones, device):
    """載入 Mamba Teacher 並凍結參數"""
    teacher = MambaTeacher(
        num_features=num_features,
        num_outs=num_speech_features,
        num_aux_outs=n_phones,
        model_size=512,
        num_layers=4,
        dropout=0.2,
    ).to(device)

    sd = torch.load(FLAGS.mamba_teacher_path, map_location=device)
    teacher.load_state_dict(sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in teacher.parameters())
    logging.info(f'✅ Mamba Teacher 載入完成: {n_params:,} params')
    return teacher


def load_transformer_teacher(num_features, num_speech_features, n_phones, device):
    """載入 Transformer Teacher 並凍結參數"""
    teacher = TransformerTeacher(
        num_features=num_features,
        num_outs=num_speech_features,
        num_aux_outs=n_phones,
        model_size=768,
        num_layers=6,
        nhead=8,
        dim_feedforward=3072,
        dropout=0.2,
    ).to(device)

    sd = torch.load(FLAGS.transformer_teacher_path, map_location=device)
    teacher.load_state_dict(sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in teacher.parameters())
    logging.info(f'✅ Transformer Teacher 載入完成: {n_params:,} params')
    return teacher


# ╔══════════════════════════════════════════════════════════════╗
# ║  Validation                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def validate(student, devset, device):
    """驗證 student 模型"""
    student.eval()
    dataloader = torch.utils.data.DataLoader(
        devset, batch_size=32, collate_fn=devset.collate_raw
    )
    losses = []
    accuracies = []
    seq_len = 200

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation', disable=None):
            X = combine_fixed_length(
                [t.to(device, non_blocking=True) for t in batch['emg']], seq_len
            )
            X_raw = combine_fixed_length(
                [t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8
            )
            sess = combine_fixed_length(
                [t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len
            )

            pred, phoneme_pred = student(X, X_raw, sess)
            loss = dtw_loss(pred, phoneme_pred, batch)
            losses.append(loss.item())

            # phoneme accuracy
            phoneme_predictions = decollate_tensor(phoneme_pred, batch['lengths'])
            phoneme_targets = batch['phonemes']
            correct = 0
            total = 0
            for pp, yp, silent in zip(phoneme_predictions, phoneme_targets, batch['silent']):
                yp = yp.to(device)
                if not silent:
                    correct += (pp.argmax(-1) == yp).sum().item()
                    total += yp.size(0)
            if total > 0:
                accuracies.append(correct / total)

    student.train()
    return np.mean(losses), np.mean(accuracies) if accuracies else 0.0


# ╔══════════════════════════════════════════════════════════════╗
# ║  Training Loop                                             ║
# ╚══════════════════════════════════════════════════════════════╝

def train_distill(trainset, devset, device):
    """主蒸餾訓練迴圈"""
    n_epochs = FLAGS.epochs
    n_phones = len(phoneme_inventory)

    student_size = FLAGS.student_model_size   # 256
    student_layers = FLAGS.student_num_layers  # 4

    # ── 載入 teachers ──────────────────────────────────────────
    logging.info('載入 Teacher 模型...')
    mamba_teacher = load_mamba_teacher(
        devset.num_features, devset.num_speech_features, n_phones, device
    )
    transformer_teacher = load_transformer_teacher(
        devset.num_features, devset.num_speech_features, n_phones, device
    )

    # ── 建立 student ───────────────────────────────────────────
    student = StudentModel(
        num_features=devset.num_features,
        num_outs=devset.num_speech_features,
        num_aux_outs=n_phones,
        model_size=student_size,
        num_layers=student_layers,
        dropout=FLAGS.dropout,
    ).to(device)

    s_params = sum(p.numel() for p in student.parameters())
    logging.info(f'Student 模型: {s_params:,} params, '
                 f'model_size={student_size}, num_layers={student_layers}')

    # ── Hint Projectors ────────────────────────────────────────
    # student hidden → teacher hidden 的投影器
    hint_proj_mamba = HintProjector(student_size, 512).to(device)
    hint_proj_transformer = HintProjector(student_size, 768).to(device)

    # ── Optimizer ──────────────────────────────────────────────
    all_params = (
        list(student.parameters()) +
        list(hint_proj_mamba.parameters()) +
        list(hint_proj_transformer.parameters())
    )
    optim = torch.optim.AdamW(all_params, weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', 0.5, patience=FLAGS.learning_rate_patience
    )

    target_lr = FLAGS.learning_rate

    def set_lr(new_lr):
        for pg in optim.param_groups:
            pg['lr'] = new_lr

    def schedule_lr(iteration):
        it = iteration + 1
        if it <= FLAGS.learning_rate_warmup:
            set_lr(it * target_lr / FLAGS.learning_rate_warmup)

    # ── DataLoader ─────────────────────────────────────────────
    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = trainset.subset(FLAGS.data_size_fraction)

    dataloader = torch.utils.data.DataLoader(
        training_subset,
        pin_memory=(device == 'cuda'),
        collate_fn=devset.collate_raw,
        num_workers=0,
        batch_sampler=SizeAwareSampler(training_subset, 64000),
    )

    seq_len = 200
    batch_idx = 0
    best_val = float('inf')

    logging.info(f'開始蒸餾訓練: {n_epochs} epochs')
    logging.info(f'  α_kd={FLAGS.alpha_kd}, α_hint={FLAGS.alpha_hint}, '
                 f'T={FLAGS.temperature}')
    logging.info(f'  teacher_weight_mamba={FLAGS.teacher_weight_mamba}')

    for epoch in range(n_epochs):
        student.train()
        epoch_losses = {
            'total': [], 'task': [], 'kd_mel': [],
            'kd_phone': [], 'hint': [],
        }

        for batch in tqdm.tqdm(dataloader, f'Epoch {epoch+1}', disable=None):
            optim.zero_grad()
            schedule_lr(batch_idx)

            # 準備資料
            X = combine_fixed_length(
                [t.to(device, non_blocking=True) for t in batch['emg']], seq_len
            )
            X_raw = combine_fixed_length(
                [t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8
            )
            sess = combine_fixed_length(
                [t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len
            )

            # ── Student forward (with hidden) ────────────────
            s_out, s_aux, s_hiddens = student(X, X_raw, sess, return_hidden=True)

            # ── Teachers forward (no grad) ────────────────────
            with torch.no_grad():
                m_out, m_aux, m_hiddens = mamba_teacher(
                    X, X_raw, sess, return_hidden=True
                )
                t_out, t_aux, t_hiddens = transformer_teacher(
                    X, X_raw, sess, return_hidden=True
                )

            # ── (1) Task loss: DTW + phoneme ─────────────────
            task_loss = dtw_loss(s_out, s_aux, batch)

            # ── (2) Output KD loss: mel ──────────────────────
            w_m = FLAGS.teacher_weight_mamba
            w_t = 1.0 - w_m

            kd_mel_mamba = distillation_loss_mse(
                s_out, m_out, FLAGS.temperature
            )
            kd_mel_transformer = distillation_loss_mse(
                s_out, t_out, FLAGS.temperature
            )
            kd_mel = w_m * kd_mel_mamba + w_t * kd_mel_transformer

            # ── (3) Output KD loss: phoneme ──────────────────
            kd_phone_mamba = distillation_loss_phoneme_kd(
                s_aux, m_aux, FLAGS.temperature
            )
            kd_phone_transformer = distillation_loss_phoneme_kd(
                s_aux, t_aux, FLAGS.temperature
            )
            kd_phone = w_m * kd_phone_mamba + w_t * kd_phone_transformer

            # ── (4) Hint loss: 中間層對齊 ─────────────────────
            # 取 student 最後一層 hidden 對齊兩個 teacher 的最後一層
            s_last_hidden = s_hiddens[-1]  # (B, T, 256)
            m_last_hidden = m_hiddens[-1]  # (B, T, 512)
            t_last_hidden = t_hiddens[-1]  # (B, T, 768)

            h_loss_mamba = hint_loss(s_last_hidden, m_last_hidden, hint_proj_mamba)
            h_loss_transformer = hint_loss(
                s_last_hidden, t_last_hidden, hint_proj_transformer
            )
            h_loss = w_m * h_loss_mamba + w_t * h_loss_transformer

            # ── 組合 total loss ───────────────────────────────
            alpha_kd = FLAGS.alpha_kd
            alpha_hint = FLAGS.alpha_hint
            alpha_task = 1.0 - alpha_kd - alpha_hint

            total_loss = (
                alpha_task * task_loss +
                alpha_kd * (kd_mel + 0.3 * kd_phone) +
                alpha_hint * h_loss
            )

            # ── Backward ─────────────────────────────────────
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optim.step()

            # ── Log ──────────────────────────────────────────
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['task'].append(task_loss.item())
            epoch_losses['kd_mel'].append(kd_mel.item())
            epoch_losses['kd_phone'].append(kd_phone.item())
            epoch_losses['hint'].append(h_loss.item())

            batch_idx += 1

        # ── Epoch 結束 ────────────────────────────────────────
        train_total = np.mean(epoch_losses['total'])
        train_task = np.mean(epoch_losses['task'])
        train_kd = np.mean(epoch_losses['kd_mel'])
        train_hint = np.mean(epoch_losses['hint'])

        val_loss, val_phoneme_acc = validate(student, devset, device)
        lr_sched.step(val_loss)

        logging.info(
            f'Epoch {epoch+1}/{n_epochs} | '
            f'total={train_total:.4f} task={train_task:.4f} '
            f'kd_mel={train_kd:.4f} hint={train_hint:.4f} | '
            f'val={val_loss:.4f} phone_acc={val_phoneme_acc*100:.1f}%'
        )

        # ── Save ─────────────────────────────────────────────
        save_path = os.path.join(FLAGS.output_directory, 'student_model.pt')
        torch.save(student.state_dict(), save_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(FLAGS.output_directory, 'student_best.pt')
            torch.save(student.state_dict(), best_path)
            logging.info(f'  💾 Best model saved (val={best_val:.4f})')

    logging.info(f'蒸餾完成! Best val loss: {best_val:.4f}')
    logging.info(f'模型儲存於: {FLAGS.output_directory}/')

    return student


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(FLAGS.output_directory, 'distill_log.txt'), 'w'
            ),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format='%(message)s',
    )

    logging.info('=' * 60)
    logging.info('SNN+Mamba Knowledge Distillation')
    logging.info('=' * 60)
    logging.info(f'Command: {sys.argv}')

    # ── 資料集 ─────────────────────────────────────────────────
    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info(f'Train: {len(trainset)} | Dev: {len(devset)}')
    logging.info(f'Features: {devset.num_features} | '
                 f'Speech features: {devset.num_speech_features}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Device: {device}')

    # ── 蒸餾訓練 ───────────────────────────────────────────────
    student = train_distill(trainset, devset, device)

    # ── 最終統計 ───────────────────────────────────────────────
    s_params = sum(p.numel() for p in student.parameters())
    s_size_mb = s_params * 4 / 1024 / 1024
    logging.info(f'\n📊 Student 模型統計:')
    logging.info(f'   參數量: {s_params:,}')
    logging.info(f'   大小 (FP32): {s_size_mb:.1f} MB')
    logging.info(f'   大小 (FP16): {s_size_mb/2:.1f} MB')
    logging.info(f'   大小 (INT8): {s_size_mb/4:.1f} MB')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
