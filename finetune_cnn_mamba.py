"""
微調腳本 — 蒸餾後的 CNN+Mamba Student 模型

載入蒸餾完成的 student 權重，用 EMG 資料以較低 learning rate
進行微調，僅使用原始 DTW + phoneme loss（不再需要 teacher）。

用法:
    python finetune_cnn_mamba.py \
        --student_checkpoint output_distill_cnn_mamba/student_best.pt \
        --output_directory output_finetune_cnn_mamba

    # 中斷後繼續:
    python finetune_cnn_mamba.py \
        --student_checkpoint output_distill_cnn_mamba/student_best.pt \
        --output_directory output_finetune_cnn_mamba --resume
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

from absl import flags
FLAGS = flags.FLAGS

# ── Flags ──────────────────────────────────────────────────────
flags.DEFINE_integer('batch_size', 16, 'training batch size')
flags.DEFINE_integer('epochs', 80, 'number of training epochs')
flags.DEFINE_float('learning_rate', 3e-4, '微調學習率 (比蒸餾時更低)')
flags.DEFINE_integer('learning_rate_patience', 5, 'lr decay patience')
flags.DEFINE_integer('learning_rate_warmup', 200, 'steps of linear warmup')
flags.DEFINE_float('l2', 1e-7, 'weight decay')
flags.DEFINE_string('output_directory', 'output_finetune_cnn_mamba', 'output directory')
flags.DEFINE_float('phoneme_loss_weight', 0.5, 'phoneme auxiliary loss weight')

flags.DEFINE_string('student_checkpoint',
                    'output_distill_cnn_mamba/student_best.pt',
                    '蒸餾完成的 student 模型權重路徑')
flags.DEFINE_integer('student_model_size', 256, 'student 隱藏層大小')
flags.DEFINE_integer('student_num_layers', 4, 'student 層數')
flags.DEFINE_float('dropout', 0.2, 'dropout')
flags.DEFINE_boolean('resume', False, '從上次中斷的 checkpoint 繼續訓練')
flags.DEFINE_boolean('freeze_conv', False, '凍結 CNN frontend 只微調 Mamba 層與輸出層')

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
# ║  模型元件 (與蒸餾腳本完全相同)                               ║
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


class StudentModel(nn.Module):
    """CNN+Mamba Student — 與蒸餾腳本完全相同的架構"""
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
        output = self.w_out(x)
        aux = self.w_aux(x) if self.has_aux_out else None

        if self.has_aux_out:
            return output, aux
        return output


# ╔══════════════════════════════════════════════════════════════╗
# ║  Loss                                                      ║
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
# ║  Fine-tune Training Loop                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def finetune(trainset, devset, device):
    n_epochs = FLAGS.epochs
    n_phones = len(phoneme_inventory)
    student_size = FLAGS.student_model_size
    student_layers = FLAGS.student_num_layers

    # ── 建立模型 ───────────────────────────────────────────────
    model = StudentModel(
        devset.num_features, devset.num_speech_features, n_phones,
        model_size=student_size, num_layers=student_layers, dropout=FLAGS.dropout,
    ).to(device)

    # ── 載入蒸餾權重 ──────────────────────────────────────────
    sd = torch.load(FLAGS.student_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    logging.info(f'✅ 載入蒸餾模型權重: {FLAGS.student_checkpoint}')

    s_params = sum(p.numel() for p in model.parameters())
    logging.info(f'模型: {s_params:,} params, d={student_size}, L={student_layers}')

    # ── 可選: 凍結 CNN frontend ───────────────────────────────
    if FLAGS.freeze_conv:
        for p in model.conv_blocks.parameters():
            p.requires_grad = False
        for p in model.w_raw_in.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'🧊 CNN frontend 已凍結, 可訓練參數: {trainable:,}')

    # ── Optimizer ──────────────────────────────────────────────
    train_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(train_params, weight_decay=FLAGS.l2)
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

    # ── Resume checkpoint ──────────────────────────────────────
    ckpt_path = os.path.join(FLAGS.output_directory, 'checkpoint.pt')
    if FLAGS.resume:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optimizer'])
            lr_sched.load_state_dict(ckpt['lr_scheduler'])
            start_epoch = ckpt['epoch'] + 1
            batch_idx = ckpt['batch_idx']
            best_val = ckpt['best_val']
            logging.info(f'⏩ 從 Epoch {start_epoch} 繼續訓練 (best_val={best_val:.4f})')
        elif os.path.exists(os.path.join(FLAGS.output_directory, 'finetuned_model.pt')):
            sd = torch.load(os.path.join(FLAGS.output_directory, 'finetuned_model.pt'),
                            map_location=device, weights_only=False)
            model.load_state_dict(sd)
            logging.info(f'⏩ 載入既有 finetuned_model.pt 權重繼續訓練 (優化器重新初始化)')
        else:
            logging.info(f'⚠️ 設定了 --resume 但未找到 checkpoint，將從蒸餾權重開始微調')

    # ── 初始 validation baseline ──────────────────────────────
    init_val = validate(model, devset, device)
    logging.info(f'📏 微調前 baseline: val={init_val:.4f}')

    logging.info(f'開始微調: {n_epochs} epochs, lr={FLAGS.learning_rate}, freeze_conv={FLAGS.freeze_conv}')

    for epoch in range(start_epoch, n_epochs):
        model.train()
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
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            optim.step()

            ep_losses.append(loss.item())
            batch_idx += 1

        # ── Epoch 結束 ────────────────────────────────────────
        val_loss = validate(model, devset, device)
        lr_sched.step(val_loss)

        train_loss = np.mean(ep_losses) if ep_losses else float('nan')
        logging.info(
            f'Epoch {epoch+1}/{n_epochs} | '
            f'train={train_loss:.4f} | '
            f'val={val_loss:.4f}'
        )

        # 儲存最新模型
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, 'finetuned_model.pt'))

        # 儲存完整 checkpoint
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
            torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, 'finetuned_best.pt'))
            logging.info(f'  💾 Best model saved (val={best_val:.4f})')

    logging.info(f'微調完成! Best val={best_val:.4f}')
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(FLAGS.output_directory, 'finetune_log.txt'),
                'a' if FLAGS.resume else 'w'
            ),
            logging.StreamHandler(),
        ],
        level=logging.INFO, format='%(message)s',
    )

    logging.info('=' * 60)
    logging.info('CNN+Mamba Fine-tuning (post-distillation)')
    logging.info('=' * 60)

    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info(f'Train: {len(trainset)} | Dev: {len(devset)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Device: {device}')

    model = finetune(trainset, devset, device)

    s_params = sum(p.numel() for p in model.parameters())
    s_mb = s_params * 4 / 1024 / 1024
    logging.info(f'\n📊 Model: {s_params:,} params | FP32={s_mb:.1f}MB | FP16={s_mb/2:.1f}MB | INT8={s_mb/4:.1f}MB')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
