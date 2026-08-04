"""
知識蒸餾訓練腳本 — 雙 Teacher → CNN+Mamba Student (無 SNN)

Teacher 1: Mamba  (models_mamba1/transduction_model/model.pt)  model_size=512, layers=4
Teacher 2: Transformer (transduction_model.pt)                 model_size=768, layers=6

Student:   CNN+Mamba, num_layers=4, model_size=256 (teacher1 的一半)

用法:
    python distill_cnn_mamba.py --output_directory output_distill_cnn_mamba
"""

import os
import sys
import random
import logging
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from read_emg import EMGDataset, SizeAwareSampler
from align import align_from_distances
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
from transformer import TransformerEncoderLayer

from absl import flags
FLAGS = flags.FLAGS

# ── Flags ──────────────────────────────────────────────────────
flags.DEFINE_integer('batch_size', 16, 'training batch size')
flags.DEFINE_integer('epochs', 80, 'number of training epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('learning_rate_patience', 5, 'lr decay patience')
flags.DEFINE_integer('learning_rate_warmup', 500, 'steps of linear warmup')
flags.DEFINE_float('l2', 1e-7, 'weight decay')
flags.DEFINE_string('output_directory', 'output_distill_cnn_mamba', 'output directory')
flags.DEFINE_float('phoneme_loss_weight', 0.5, 'phoneme auxiliary loss weight')

flags.DEFINE_string('mamba_teacher_path',
                    'models_mamba1/transduction_model/model.pt',
                    'Mamba teacher checkpoint')
flags.DEFINE_string('transformer_teacher_path',
                    'transduction_model.pt',
                    'Transformer teacher checkpoint')

flags.DEFINE_float('alpha_kd', 0.5, '蒸餾 loss 權重')
flags.DEFINE_float('alpha_hint', 0.2, '中間層 hint loss 權重')
flags.DEFINE_float('temperature', 2.0, '蒸餾溫度 (降低可讓 soft target 更銳利)')
flags.DEFINE_float('teacher_weight_mamba', 0.3, 'Mamba teacher 權重')
flags.DEFINE_integer('student_model_size', 256, 'student 隱藏層大小')
flags.DEFINE_integer('student_num_layers', 4, 'student 層數')
flags.DEFINE_float('dropout', 0.2, 'dropout')
flags.DEFINE_integer('stage1_epochs', 15, '階段一 (特徵對齊為主) 的 epoch 數，之後切換到階段二 (輸出對齊為主)')
flags.DEFINE_boolean('resume', False, '從上次中斷的 checkpoint 繼續訓練')

for _name, _default, _help in [
    ('model_size', 512, 'teacher model hidden size'),
    ('num_layers', 4, 'teacher num layers'),
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
# ║  共用元件                                                   ║
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
    """標準 Mamba 區塊 — 與 teacher 相同結構"""
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


# ╔══════════════════════════════════════════════════════════════╗
# ║  Student: CNN + Mamba (小型版)                               ║
# ╚══════════════════════════════════════════════════════════════╝

class StudentModel(nn.Module):
    """
    CNN+Mamba Student — 與 teacher 相同架構, 隱藏維度減半

    差異僅在 model_size: teacher=512 → student=256
    """
    def __init__(self, num_features, num_outs, num_aux_outs=None,
                 model_size=256, num_layers=4, dropout=0.2):
        super().__init__()
        self.model_size = model_size

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

    def forward(self, x_feat, x_raw, session_ids, return_hidden=False):
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
        for layer in self.mamba_layers:
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
# ║  Teacher Wrappers                                          ║
# ╚══════════════════════════════════════════════════════════════╝

class MambaTeacher(nn.Module):
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
            MambaBlock(d_model=model_size, dropout=dropout)
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
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=model_size, nhead=nhead,
                relative_positional=True, relative_positional_distance=100,
                dim_feedforward=dim_feedforward, dropout=dropout,
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
        x = x.transpose(0, 1)
        hiddens = []
        for layer in self.transformer:
            x = layer(x)
            if return_hidden:
                hiddens.append(x.transpose(0, 1))  # (B, T, D)
        x = x.transpose(0, 1)
        out = self.w_out(x)
        aux = self.w_aux(x) if self.has_aux_out else None
        if return_hidden:
            return out, aux, hiddens
        if self.has_aux_out:
            return out, aux
        return out


# ╔══════════════════════════════════════════════════════════════╗
# ║  Projector & Loss                                          ║
# ╚══════════════════════════════════════════════════════════════╝

class HintProjector(nn.Module):
    def __init__(self, d_student, d_teacher):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_student, d_teacher),
            nn.ReLU(),
            nn.Linear(d_teacher, d_teacher),
        )

    def forward(self, x):
        return self.proj(x)


def distillation_loss_mse(student_logits, teacher_logits):
    return F.mse_loss(student_logits, teacher_logits)


def distillation_loss_phoneme_kd(student_phoneme, teacher_phoneme, temperature):
    if student_phoneme is None or teacher_phoneme is None:
        dev = (student_phoneme or teacher_phoneme)
        if dev is not None:
            return torch.tensor(0.0, device=dev.device)
        return torch.tensor(0.0)
    T = temperature
    s_log = F.log_softmax(student_phoneme / T, dim=-1)
    t_prob = F.softmax(teacher_phoneme / T, dim=-1)
    return F.kl_div(s_log, t_prob, reduction='batchmean') * (T ** 2)


def hint_loss(student_hidden, teacher_hidden, projector):
    return F.mse_loss(projector(student_hidden), teacher_hidden.detach())


def multi_layer_hint_loss(s_hiddens, t_hiddens, projectors):
    """
    多層中間特徵對齊: 將 student 的每一層 hidden 對齊到
    teacher 對應層的 hidden。若 teacher 層數不同於 student，
    使用等間距採樣對齊。
    """
    n_student = len(s_hiddens)
    n_teacher = len(t_hiddens)
    total_loss = 0.0
    for i in range(n_student):
        # 等間距映射: student layer i → teacher layer j
        j = min(int(i * n_teacher / n_student), n_teacher - 1)
        total_loss = total_loss + hint_loss(s_hiddens[i], t_hiddens[j], projectors[i])
    return total_loss / n_student


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
# ║  Teacher 載入                                              ║
# ╚══════════════════════════════════════════════════════════════╝

def load_mamba_teacher(num_features, num_speech_features, n_phones, device):
    teacher = MambaTeacher(num_features, num_speech_features, n_phones,
                           model_size=512, num_layers=4).to(device)
    sd = torch.load(FLAGS.mamba_teacher_path, map_location=device, weights_only=False)
    teacher.load_state_dict(sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logging.info(f'✅ Mamba Teacher: {sum(p.numel() for p in teacher.parameters()):,} params')
    return teacher


def load_transformer_teacher(num_features, num_speech_features, n_phones, device):
    teacher = TransformerTeacher(num_features, num_speech_features, n_phones,
                                 model_size=768, num_layers=6).to(device)
    sd = torch.load(FLAGS.transformer_teacher_path, map_location=device, weights_only=False)
    teacher.load_state_dict(sd, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logging.info(f'✅ Transformer Teacher: {sum(p.numel() for p in teacher.parameters()):,} params')
    return teacher


# ╔══════════════════════════════════════════════════════════════╗
# ║  Validation                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def validate(student, devset, device):
    student.eval()
    dataloader = torch.utils.data.DataLoader(devset, batch_size=32, collate_fn=devset.collate_raw)
    losses = []
    accuracies = []
    seq_len = 200

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation', disable=None):
            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            pred, phoneme_pred = student(X, X_raw, sess)
            loss = dtw_loss(pred, phoneme_pred, batch)
            losses.append(loss.item())

            preds_list = decollate_tensor(phoneme_pred, batch['lengths'])
            correct, total = 0, 0
            for pp, yp, sil in zip(preds_list, batch['phonemes'], batch['silent']):
                if not sil:
                    yp = yp.to(device)
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
    n_epochs = FLAGS.epochs
    n_phones = len(phoneme_inventory)
    student_size = FLAGS.student_model_size
    student_layers = FLAGS.student_num_layers

    # Teachers
    mamba_teacher = load_mamba_teacher(devset.num_features, devset.num_speech_features, n_phones, device)
    transformer_teacher = load_transformer_teacher(devset.num_features, devset.num_speech_features, n_phones, device)

    # Student
    student = StudentModel(
        devset.num_features, devset.num_speech_features, n_phones,
        model_size=student_size, num_layers=student_layers, dropout=FLAGS.dropout,
    ).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    logging.info(f'Student CNN+Mamba: {s_params:,} params, d={student_size}, L={student_layers}')

    # Hint projectors — 每層各一個 (多層對齊)
    hint_projs_mamba = nn.ModuleList([
        HintProjector(student_size, 512) for _ in range(student_layers)
    ]).to(device)
    hint_projs_transformer = nn.ModuleList([
        HintProjector(student_size, 768) for _ in range(student_layers)
    ]).to(device)

    # Optimizer
    all_params = (
        list(student.parameters()) +
        list(hint_projs_mamba.parameters()) +
        list(hint_projs_transformer.parameters())
    )
    optim = torch.optim.AdamW(all_params, weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=FLAGS.learning_rate_patience)
    target_lr = FLAGS.learning_rate

    def set_lr(lr):
        for pg in optim.param_groups:
            pg['lr'] = lr

    def schedule_lr(it):
        if it + 1 <= FLAGS.learning_rate_warmup:
            set_lr((it + 1) * target_lr / FLAGS.learning_rate_warmup)

    # DataLoader
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

    # ── Resume checkpoint ──────────────────────────────────────
    ckpt_path = os.path.join(FLAGS.output_directory, 'checkpoint.pt')
    student_model_path = os.path.join(FLAGS.output_directory, 'student_model.pt')

    if FLAGS.resume:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            student.load_state_dict(ckpt['student'])
            if 'hint_projs_mamba' in ckpt:
                hint_projs_mamba.load_state_dict(ckpt['hint_projs_mamba'])
                hint_projs_transformer.load_state_dict(ckpt['hint_projs_transformer'])
            elif 'hint_proj_mamba' in ckpt:
                # 舊版 checkpoint 相容 (單層 projector)
                logging.info('⚠️ 偵測到舊版 checkpoint (單層 hint projector)，只載入 student 權重')
            optim.load_state_dict(ckpt['optimizer'])
            lr_sched.load_state_dict(ckpt['lr_scheduler'])
            start_epoch = ckpt['epoch'] + 1
            batch_idx = ckpt['batch_idx']
            best_val = ckpt['best_val']
            logging.info(f'⏩ 從 Epoch {start_epoch} 繼續訓練 (best_val={best_val:.4f})')
        elif os.path.exists(student_model_path):
            # 只有 student_model.pt (舊版純權重)，優化器等狀態重新初始化
            sd = torch.load(student_model_path, map_location=device, weights_only=False)
            student.load_state_dict(sd)
            logging.info(f'⏩ 未找到完整 checkpoint.pt，但已載入已存在之 {student_model_path} 權重繼續訓練 (優化器與 Epoch 將重新初始化)')
        else:
            logging.info(f'⚠️ 設定了 --resume 但未找到任何 checkpoint/權重，將從頭開始訓練')

    logging.info(f'蒸餾訓練: {n_epochs} epochs, α_kd={FLAGS.alpha_kd}, α_hint={FLAGS.alpha_hint}, T={FLAGS.temperature}')
    logging.info(f'階段式蒸餾: 階段一 (特徵對齊) = Epoch 1~{FLAGS.stage1_epochs}, 階段二 (輸出對齊) = Epoch {FLAGS.stage1_epochs+1}~{n_epochs}')

    for epoch in range(start_epoch, n_epochs):
        student.train()
        ep = {'total': [], 'task': [], 'kd': [], 'hint': []}

        # ── 階段式蒸餾: 動態調整 loss 權重 ─────────────────────
        if epoch < FLAGS.stage1_epochs:
            # 階段一: 著重特徵空間對齊 (hint 為主)
            alpha_hint = 0.4
            alpha_kd = 0.3
            alpha_task = 0.3
        else:
            # 階段二: 著重輸出品質 (task + kd 為主)
            alpha_hint = 0.05
            alpha_kd = 0.55
            alpha_task = 0.4

        for batch in tqdm.tqdm(dataloader, f'Epoch {epoch+1}', disable=None):
            optim.zero_grad()
            schedule_lr(batch_idx)

            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            # Student
            s_out, s_aux, s_hiddens = student(X, X_raw, sess, return_hidden=True)

            # Teachers
            with torch.no_grad():
                m_out, m_aux, m_hiddens = mamba_teacher(X, X_raw, sess, return_hidden=True)
                t_out, t_aux, t_hiddens = transformer_teacher(X, X_raw, sess, return_hidden=True)

            # (1) Task loss
            task_loss = dtw_loss(s_out, s_aux, batch)

            # (2) Output KD
            w_m = FLAGS.teacher_weight_mamba
            w_t = 1.0 - w_m
            kd_mel = w_m * distillation_loss_mse(s_out, m_out) + w_t * distillation_loss_mse(s_out, t_out)
            kd_phone = w_m * distillation_loss_phoneme_kd(s_aux, m_aux, FLAGS.temperature) \
                     + w_t * distillation_loss_phoneme_kd(s_aux, t_aux, FLAGS.temperature)
            kd_total = kd_mel + 0.3 * kd_phone

            # (3) Multi-layer hint loss (多層中間特徵對齊)
            h_loss_mamba = multi_layer_hint_loss(s_hiddens, m_hiddens, hint_projs_mamba)
            h_loss_transformer = multi_layer_hint_loss(s_hiddens, t_hiddens, hint_projs_transformer)
            h_loss = w_m * h_loss_mamba + w_t * h_loss_transformer

            # Total (使用階段式動態權重)
            total_loss = alpha_task * task_loss + alpha_kd * kd_total + alpha_hint * h_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optim.zero_grad()
                batch_idx += 1
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optim.step()

            ep['total'].append(total_loss.item())
            ep['task'].append(task_loss.item())
            ep['kd'].append(kd_mel.item())
            ep['hint'].append(h_loss.item())
            batch_idx += 1

        # Epoch end
        stage_name = '特徵對齊' if epoch < FLAGS.stage1_epochs else '輸出對齊'
        val_loss, val_acc = validate(student, devset, device)
        lr_sched.step(val_loss)
        logging.info(
            f'Epoch {epoch+1}/{n_epochs} [{stage_name}] | '
            f'total={np.mean(ep["total"]):.4f} task={np.mean(ep["task"]):.4f} '
            f'kd={np.mean(ep["kd"]):.4f} hint={np.mean(ep["hint"]):.4f} | '
            f'val={val_loss:.4f} phone_acc={val_acc*100:.1f}%'
        )

        torch.save(student.state_dict(), os.path.join(FLAGS.output_directory, 'student_model.pt'))

        # 儲存完整 checkpoint (可用於斷點續訓)
        torch.save({
            'student': student.state_dict(),
            'hint_projs_mamba': hint_projs_mamba.state_dict(),
            'hint_projs_transformer': hint_projs_transformer.state_dict(),
            'optimizer': optim.state_dict(),
            'lr_scheduler': lr_sched.state_dict(),
            'epoch': epoch,
            'batch_idx': batch_idx,
            'best_val': best_val,
        }, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), os.path.join(FLAGS.output_directory, 'student_best.pt'))
            logging.info(f'  💾 Best model saved (val={best_val:.4f})')

    logging.info(f'蒸餾完成! Best val={best_val:.4f}')
    return student


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'distill_log.txt'), 'a' if FLAGS.resume else 'w'),
            logging.StreamHandler(),
        ],
        level=logging.INFO, format='%(message)s',
    )

    logging.info('=' * 60)
    logging.info('CNN+Mamba Knowledge Distillation (no SNN)')
    logging.info('=' * 60)

    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info(f'Train: {len(trainset)} | Dev: {len(devset)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Device: {device}')

    student = train_distill(trainset, devset, device)

    s_params = sum(p.numel() for p in student.parameters())
    s_mb = s_params * 4 / 1024 / 1024
    logging.info(f'\n📊 Student: {s_params:,} params | FP32={s_mb:.1f}MB | FP16={s_mb/2:.1f}MB | INT8={s_mb/4:.1f}MB')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
