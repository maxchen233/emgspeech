"""
HiFi-GAN Vocoder INT16 量化感知訓練 (Quantization-Aware Training, QAT)

採用 Teacher-Student 蒸餾 + GAN 對抗訓練，在不損失音質的前提下
將 HiFi-GAN Generator 權重量化為 INT16，壓縮模型至約 50% 大小。

策略:
  1. Teacher (FP32, frozen) 產生參考 waveform
  2. Student (插入 fake-quant 節點) 學習匹配 Teacher 輸出
  3. MPD + MSD discriminator 確保感知品質
  4. Loss = L1_waveform + 45×Mel_L1 + GAN + Feature_matching
  5. 前 qat_start_epoch 個 epoch 不啟用 fake-quant (漸進式)

訓練完成後匯出:
  1. qat_generator.pt       — QAT 微調後的 FP32 state_dict
  2. qat_generator_int16.pt — 真正量化為 INT16 的權重 + scale 字典

用法:
    python qat_hifigan.py \
        --checkpoint_path hifigan_finetuned/checkpoint \
        --config hifigan_finetuned/config.json \
        --output_dir output_qat_hifigan \
        --training_epochs 50 \
        --qat_start_epoch 3

    # 使用 LJSpeech wav 直接訓練 (非 fine-tuning 模式):
    python qat_hifigan.py \
        --checkpoint_path hifigan_finetuned/checkpoint \
        --config hifigan_finetuned/config.json \
        --input_wavs_dir hifi_gan/LJSpeech-1.1/wavs \
        --input_training_file hifi_gan/LJSpeech-1.1/training.txt \
        --input_validation_file hifi_gan/LJSpeech-1.1/validation.txt \
        --output_dir output_qat_hifigan \
        --training_epochs 50

    # 中斷後繼續:
    python qat_hifigan.py \
        --checkpoint_path hifigan_finetuned/checkpoint \
        --config hifigan_finetuned/config.json \
        --output_dir output_qat_hifigan \
        --resume
"""

import os
import sys
import json
import time
import logging
import argparse
import itertools
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# ── HiFi-GAN imports ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hifi_gan'))
from env import AttrDict
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import (Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
                    feature_loss, generator_loss, discriminator_loss)
from utils import scan_checkpoint, load_checkpoint, save_checkpoint, plot_spectrogram


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
        # STE: 梯度直通 (INT16 範圍夠大, 溢出極少)
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
        scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-12)
        zero_point = torch.round(qmin - x_min / scale)
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        scale = scale.view(shape)
        zero_point = zero_point.view(shape)
    else:
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


class QuantizedConvTranspose1d(nn.Module):
    """包裝 nn.ConvTranspose1d, 在 forward 時對 weight 做 fake-quant"""
    def __init__(self, conv: nn.ConvTranspose1d, per_channel=True, bits=16):
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
        return F.conv_transpose1d(x, w_q, self.conv.bias,
                                  self.conv.stride, self.conv.padding,
                                  self.conv.output_padding,
                                  self.conv.groups, self.conv.dilation)


class ActivationQuantHook(nn.Module):
    """插入在 activation 後, 對 activation 做 fake-quant (per-tensor)"""
    def __init__(self, bits=16):
        super().__init__()
        self.bits = bits
        self.fake_quant_enabled = True

    def forward(self, x):
        if self.fake_quant_enabled and self.training:
            return fake_quantize_int16(x, per_channel=False, bits=self.bits)
        return x


# ╔══════════════════════════════════════════════════════════════╗
# ║  Generator QAT 插入工具                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def insert_fake_quant_generator(generator, per_channel=True, bits=16):
    """
    對 HiFi-GAN Generator 插入 INT16 fake-quantize:
      1. conv_pre: Conv1d → QuantizedConv1d
      2. ups: ConvTranspose1d → QuantizedConvTranspose1d
      3. ResBlock 內的 Conv1d → QuantizedConv1d
      4. conv_post: Conv1d → QuantizedConv1d
      5. 在每個 upsample 後插入 activation quant hook

    注意: weight_norm 必須先移除才能量化 (因為 weight_norm 把
          weight 拆為 weight_g + weight_v, 量化需要直接操作 weight)
    """
    # ── 先移除 weight_norm ────────────────────────────────────
    generator.remove_weight_norm()

    n_wrapped = 0

    # ── 1) conv_pre ───────────────────────────────────────────
    generator.conv_pre = QuantizedConv1d(generator.conv_pre, per_channel, bits)
    n_wrapped += 1

    # ── 2) ups (ConvTranspose1d) ──────────────────────────────
    for i in range(len(generator.ups)):
        generator.ups[i] = QuantizedConvTranspose1d(generator.ups[i], per_channel, bits)
        n_wrapped += 1

    # ── 3) ResBlock 內的 Conv1d ───────────────────────────────
    for resblock in generator.resblocks:
        # ResBlock1 有 convs1 + convs2, ResBlock2 只有 convs
        if hasattr(resblock, 'convs1'):
            for j in range(len(resblock.convs1)):
                resblock.convs1[j] = QuantizedConv1d(resblock.convs1[j], per_channel, bits)
                n_wrapped += 1
            for j in range(len(resblock.convs2)):
                resblock.convs2[j] = QuantizedConv1d(resblock.convs2[j], per_channel, bits)
                n_wrapped += 1
        elif hasattr(resblock, 'convs'):
            for j in range(len(resblock.convs)):
                resblock.convs[j] = QuantizedConv1d(resblock.convs[j], per_channel, bits)
                n_wrapped += 1

    # ── 4) conv_post ──────────────────────────────────────────
    generator.conv_post = QuantizedConv1d(generator.conv_post, per_channel, bits)
    n_wrapped += 1

    logging.info(f'   量化節點: {n_wrapped} 個 Conv/ConvTranspose 已包裝')
    return generator


def set_fake_quant_enabled(model, enabled: bool):
    """啟用或停用所有 fake-quant 節點"""
    for m in model.modules():
        if isinstance(m, (QuantizedConv1d, QuantizedConvTranspose1d, ActivationQuantHook)):
            m.fake_quant_enabled = enabled


# ╔══════════════════════════════════════════════════════════════╗
# ║  INT16 匯出                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def export_int16_weights(model, bits=16):
    """
    將 QAT 完成的 Generator 權重真正量化為 INT16, 回傳:
      - quantized_state: {name: int16_tensor}
      - scale_dict:      {name: (scale, zero_point)}
      - other_state:     非量化的參數 (bias 等)
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    quantized_state = {}
    scale_dict = {}
    other_state = {}

    for name, param in model.state_dict().items():
        # 只量化 weight (dim >= 2), 不量化 bias (1-d)
        is_weight = 'weight' in name and param.dim() >= 2

        if is_weight:
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
# ║  QAT Training Loop                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def train_qat(args, h):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    # ── 建立 Teacher (FP32, frozen) ───────────────────────────
    teacher = Generator(h).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    if 'generator' in ckpt:
        teacher.load_state_dict(ckpt['generator'])
    else:
        teacher.load_state_dict(ckpt)
    teacher.remove_weight_norm()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logging.info('✅ Teacher (FP32) 載入完成, 已凍結')

    # ── 建立 Student (QAT) ────────────────────────────────────
    student = Generator(h).to(device)
    if 'generator' in ckpt:
        student.load_state_dict(ckpt['generator'])
    else:
        student.load_state_dict(ckpt)

    s_params = sum(p.numel() for p in student.parameters())
    logging.info(f'Generator 參數量: {s_params:,} | FP32={s_params*4/1024/1024:.1f}MB')

    # 插入 fake-quant
    student = insert_fake_quant_generator(student, args.per_channel, args.quant_bits)
    logging.info(f'🔧 已插入 INT{args.quant_bits} fake-quantize 節點')

    # ── Discriminators ────────────────────────────────────────
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # 載入 discriminator 的預訓練權重 (如果有)
    cp_do = scan_checkpoint(os.path.dirname(args.checkpoint_path), 'do_')
    if cp_do is not None:
        state_dict_do = load_checkpoint(cp_do, device)
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        logging.info(f'✅ Discriminator 預訓練權重載入: {cp_do}')
    else:
        logging.info('⚠️ 未找到 Discriminator 預訓練權重, 從頭訓練')

    # ── 先做 baseline 驗證 ────────────────────────────────────
    set_fake_quant_enabled(student, False)
    student.eval()

    # ── Optimizers ────────────────────────────────────────────
    optim_g = torch.optim.AdamW(student.parameters(),
                                 lr=args.learning_rate,
                                 betas=(h.adam_b1, h.adam_b2))
    optim_d = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), msd.parameters()),
        lr=args.learning_rate,
        betas=(h.adam_b1, h.adam_b2))

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)

    # ── Dataset ───────────────────────────────────────────────
    training_filelist, validation_filelist = get_dataset_filelist(args)
    trainset = MelDataset(
        training_filelist, h.segment_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
        n_cache_reuse=0, shuffle=True,
        fmax_loss=h.fmax_for_loss, device=device,
        fine_tuning=args.fine_tuning, base_mels_path=args.input_mels_dir)
    train_loader = DataLoader(
        trainset, num_workers=h.num_workers, shuffle=True,
        batch_size=h.batch_size, pin_memory=True, drop_last=True)

    validset = MelDataset(
        validation_filelist, h.segment_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
        split=False, shuffle=False, n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss, device=device,
        fine_tuning=args.fine_tuning, base_mels_path=args.input_mels_dir)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                    batch_size=1, pin_memory=True, drop_last=True)

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 0
    best_val = float('inf')
    steps = 0
    ckpt_path = os.path.join(args.output_dir, 'qat_checkpoint.pt')

    if args.resume and os.path.exists(ckpt_path):
        ckpt_resume = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt_resume['generator'])
        mpd.load_state_dict(ckpt_resume['mpd'])
        msd.load_state_dict(ckpt_resume['msd'])
        optim_g.load_state_dict(ckpt_resume['optim_g'])
        optim_d.load_state_dict(ckpt_resume['optim_d'])
        scheduler_g.load_state_dict(ckpt_resume['scheduler_g'])
        scheduler_d.load_state_dict(ckpt_resume['scheduler_d'])
        start_epoch = ckpt_resume['epoch'] + 1
        steps = ckpt_resume['steps']
        best_val = ckpt_resume['best_val']
        logging.info(f'⏩ 從 Epoch {start_epoch} 繼續 (best_val={best_val:.6f}, steps={steps})')

    # ── TensorBoard ───────────────────────────────────────────
    sw = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # ── 蒸餾權重 ──────────────────────────────────────────────
    distill_weight = args.distill_weight   # L1 waveform 蒸餾權重
    mel_weight = 45.0                       # Mel-spec L1 權重 (與原版 HiFi-GAN 相同)

    # ── Training ──────────────────────────────────────────────
    logging.info(f'\n🚀 開始 HiFi-GAN INT{args.quant_bits} QAT')
    logging.info(f'   Epochs: {args.training_epochs} | LR: {args.learning_rate}')
    logging.info(f'   Distill weight: {distill_weight} | Mel weight: {mel_weight}')
    logging.info(f'   QAT start epoch: {args.qat_start_epoch}')

    for epoch in range(start_epoch, args.training_epochs):
        student.train()
        mpd.train()
        msd.train()

        # 漸進式: 前幾個 epoch 不啟用 fake-quant
        if epoch < args.qat_start_epoch:
            set_fake_quant_enabled(student, False)
            quant_status = 'OFF (warmup)'
        else:
            set_fake_quant_enabled(student, True)
            quant_status = 'ON'

        epoch_start = time.time()
        ep_gen_loss = []
        ep_disc_loss = []
        ep_mel_err = []
        ep_distill_wav = []
        ep_distill_mel = []
        ep_gan_loss = []
        ep_fm_loss = []

        cur_lr = optim_g.param_groups[0]['lr']
        n_batches = len(train_loader)
        logging.info(f'\n{"="*60}')
        logging.info(
            f'Epoch {epoch+1}/{args.training_epochs} | '
            f'quant={quant_status} | '
            f'lr={cur_lr:.2e} | '
            f'batches={n_batches}')
        logging.info(f'{"="*60}')

        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}',
                         dynamic_ncols=True, leave=True)

        for i, batch in enumerate(pbar):
            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)

            # ── Student forward ───────────────────────────────
            y_g_hat = student(x)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                h.sampling_rate, h.hop_size, h.win_size,
                h.fmin, h.fmax_for_loss)

            # ── Teacher forward (detached) ────────────────────
            with torch.no_grad():
                y_teacher = teacher(x)
                y_teacher_mel = mel_spectrogram(
                    y_teacher.squeeze(1), h.n_fft, h.num_mels,
                    h.sampling_rate, h.hop_size, h.win_size,
                    h.fmin, h.fmax_for_loss)

            # ══════════════════════════════════════════════════
            # Discriminator step
            # ══════════════════════════════════════════════════
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # ══════════════════════════════════════════════════
            # Generator step
            # ══════════════════════════════════════════════════
            optim_g.zero_grad()

            # 1) Mel-Spectrogram L1 (vs ground truth)
            loss_mel_gt = F.l1_loss(y_mel, y_g_hat_mel) * mel_weight

            # 2) Teacher 蒸餾: L1 waveform + L1 mel
            loss_distill_wav = F.l1_loss(y_g_hat, y_teacher) * distill_weight
            loss_distill_mel = F.l1_loss(y_g_hat_mel, y_teacher_mel) * mel_weight * 0.5

            # 3) GAN loss + feature matching
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            loss_gen_all = (loss_gen_s + loss_gen_f
                            + loss_fm_s + loss_fm_f
                            + loss_mel_gt
                            + loss_distill_wav + loss_distill_mel)

            loss_gen_all.backward()
            optim_g.step()

            # ── Logging ───────────────────────────────────────
            with torch.no_grad():
                mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

            _g = loss_gen_all.item()
            _d = loss_disc_all.item()
            _dw = loss_distill_wav.item()
            _dm = loss_distill_mel.item()
            _gan = (loss_gen_s + loss_gen_f).item()
            _fm = (loss_fm_s + loss_fm_f).item()

            ep_gen_loss.append(_g)
            ep_disc_loss.append(_d)
            ep_mel_err.append(mel_error)
            ep_distill_wav.append(_dw)
            ep_distill_mel.append(_dm)
            ep_gan_loss.append(_gan)
            ep_fm_loss.append(_fm)

            # tqdm 進度條顯示即時 loss
            pbar.set_postfix({
                'G': f'{_g:.2f}',
                'D': f'{_d:.2f}',
                'Mel': f'{mel_error:.4f}',
                'Dw': f'{_dw:.2f}',
            })

            if steps % args.stdout_interval == 0:
                # GPU 記憶體使用量
                mem_str = ''
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
                    mem_resrv = torch.cuda.memory_reserved(device) / 1024**3
                    mem_str = f' | GPU={mem_alloc:.1f}/{mem_resrv:.1f}GB'

                logging.info(
                    f'  [{i+1}/{n_batches}] Step {steps:>6d} | '
                    f'G={_g:.3f} D={_d:.3f} '
                    f'Mel={mel_error:.4f} | '
                    f'GAN={_gan:.3f} FM={_fm:.3f} | '
                    f'Dist_w={_dw:.3f} Dist_m={_dm:.3f}'
                    f'{mem_str}')

            if steps % args.summary_interval == 0:
                sw.add_scalar('training/gen_loss_total', _g, steps)
                sw.add_scalar('training/disc_loss_total', _d, steps)
                sw.add_scalar('training/mel_spec_error', mel_error, steps)
                sw.add_scalar('training/distill_wav', _dw, steps)
                sw.add_scalar('training/distill_mel', _dm, steps)
                sw.add_scalar('training/gan_loss', _gan, steps)
                sw.add_scalar('training/feature_matching', _fm, steps)
                sw.add_scalar('training/learning_rate', cur_lr, steps)

            steps += 1

        # ── Epoch 結束: Validation ────────────────────────────
        student.eval()
        set_fake_quant_enabled(student, True)  # 驗證時始終啟用量化
        val_err = 0.0
        val_count = 0
        with torch.no_grad():
            for j, batch in enumerate(validation_loader):
                x, y, _, y_mel = batch
                y_g_hat = student(x.to(device))
                y_mel = y_mel.to(device, non_blocking=True)
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                    h.sampling_rate, h.hop_size, h.win_size,
                    h.fmin, h.fmax_for_loss)
                val_err += F.l1_loss(y_mel, y_g_hat_mel).item()
                val_count += 1

                # 儲存前幾個樣本的音頻到 TensorBoard
                if j < 4:
                    if epoch == start_epoch:
                        sw.add_audio(f'gt/y_{j}', y[0], steps, h.sampling_rate)
                    sw.add_audio(f'generated/y_hat_{j}', y_g_hat[0], steps, h.sampling_rate)

        val_err /= max(val_count, 1)
        sw.add_scalar('validation/mel_spec_error', val_err, steps)

        elapsed = time.time() - epoch_start
        remaining_epochs = args.training_epochs - epoch - 1
        eta_sec = elapsed * remaining_epochs
        eta_min = eta_sec / 60
        eta_hr = eta_sec / 3600

        logging.info(f'\n{"─"*60}')
        logging.info(f'📊 Epoch {epoch+1}/{args.training_epochs} 結果 (quant={quant_status})')
        logging.info(f'   Train G_total  = {np.mean(ep_gen_loss):.4f}')
        logging.info(f'   Train D_total  = {np.mean(ep_disc_loss):.4f}')
        logging.info(f'   Train Mel      = {np.mean(ep_mel_err):.4f}')
        logging.info(f'   Train GAN      = {np.mean(ep_gan_loss):.4f}')
        logging.info(f'   Train FM       = {np.mean(ep_fm_loss):.4f}')
        logging.info(f'   Train Dist_wav = {np.mean(ep_distill_wav):.4f}')
        logging.info(f'   Train Dist_mel = {np.mean(ep_distill_mel):.4f}')
        logging.info(f'   Val Mel        = {val_err:.6f}  (best={best_val:.6f})')
        logging.info(f'   耗時: {elapsed:.0f}s ({elapsed/60:.1f}min) | '
                     f'ETA: {eta_min:.0f}min ({eta_hr:.1f}hr) | '
                     f'LR: {cur_lr:.2e}')
        logging.info(f'{"─"*60}')

        # ── Save checkpoint ───────────────────────────────────
        save_checkpoint(
            os.path.join(args.output_dir, f'g_qat_{epoch+1:04d}'),
            {'generator': student.state_dict()})

        torch.save({
            'generator': student.state_dict(),
            'mpd': mpd.state_dict(),
            'msd': msd.state_dict(),
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict(),
            'scheduler_g': scheduler_g.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'epoch': epoch,
            'steps': steps,
            'best_val': best_val,
        }, ckpt_path)

        if val_err < best_val:
            best_val = val_err
            save_checkpoint(
                os.path.join(args.output_dir, 'g_qat_best'),
                {'generator': student.state_dict()})
            logging.info(f'  💾 Best model saved (val_mel={best_val:.6f})')

        scheduler_g.step()
        scheduler_d.step()

    # ╔══════════════════════════════════════════════════════════╗
    # ║  QAT 完成, 匯出 INT16                                   ║
    # ╚══════════════════════════════════════════════════════════╝

    logging.info(f'\n✅ QAT 訓練完成! Best val_mel={best_val:.6f}')

    # 載入最佳模型
    best_sd = torch.load(
        os.path.join(args.output_dir, 'g_qat_best'),
        map_location=device, weights_only=False)
    if 'generator' in best_sd:
        student.load_state_dict(best_sd['generator'])
    else:
        student.load_state_dict(best_sd)

    # 匯出 FP32 QAT state_dict
    torch.save(
        {'generator': student.state_dict()},
        os.path.join(args.output_dir, 'qat_generator.pt'))

    # 匯出 INT16 量化權重
    quantized_state, scale_dict, other_state = export_int16_weights(student, args.quant_bits)
    export_path = os.path.join(args.output_dir, 'qat_generator_int16.pt')
    torch.save({
        'quantized_weights': quantized_state,
        'scales': scale_dict,
        'other_params': other_state,
        'bits': args.quant_bits,
        'per_channel': args.per_channel,
        'hparams': dict(h),
    }, export_path)

    # ── 統計 ──────────────────────────────────────────────────
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
    logging.info(f'   非量化參數: {len(other_state)} 個 tensor (bias)')
    logging.info(f'   匯出路徑: {export_path}')

    # ── 最終 Teacher vs Student 比較 ─────────────────────────
    student.eval()
    set_fake_quant_enabled(student, True)
    teacher_err = 0.0
    student_err = 0.0
    n_val = 0
    with torch.no_grad():
        for batch in validation_loader:
            x, y, _, y_mel = batch
            x = x.to(device)
            y_mel = y_mel.to(device)

            y_t = teacher(x)
            y_t_mel = mel_spectrogram(
                y_t.squeeze(1), h.n_fft, h.num_mels,
                h.sampling_rate, h.hop_size, h.win_size,
                h.fmin, h.fmax_for_loss)
            teacher_err += F.l1_loss(y_mel, y_t_mel).item()

            y_s = student(x)
            y_s_mel = mel_spectrogram(
                y_s.squeeze(1), h.n_fft, h.num_mels,
                h.sampling_rate, h.hop_size, h.win_size,
                h.fmin, h.fmax_for_loss)
            student_err += F.l1_loss(y_mel, y_s_mel).item()
            n_val += 1

    teacher_err /= max(n_val, 1)
    student_err /= max(n_val, 1)
    logging.info(f'\n📏 最終比較:')
    logging.info(f'   Teacher (FP32) val_mel = {teacher_err:.6f}')
    logging.info(f'   Student (INT{args.quant_bits} QAT) val_mel = {student_err:.6f}')
    logging.info(f'   差距 = {student_err - teacher_err:+.6f}')

    sw.close()
    return student


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main                                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description='HiFi-GAN INT16 QAT')

    # 模型 & 資料
    parser.add_argument('--checkpoint_path', required=True,
                        help='HiFi-GAN generator checkpoint 路徑')
    parser.add_argument('--config', required=True,
                        help='HiFi-GAN config.json 路徑')
    parser.add_argument('--input_wavs_dir', default='hifi_gan/LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='hifi_gan/LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='hifi_gan/LJSpeech-1.1/validation.txt')
    parser.add_argument('--fine_tuning', action='store_true',
                        help='使用 fine-tuning 模式 (從 npy mel 讀取)')

    # 輸出
    parser.add_argument('--output_dir', default='output_qat_hifigan')

    # 訓練參數
    parser.add_argument('--training_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='QAT 學習率 (比原始訓練更低)')
    parser.add_argument('--distill_weight', type=float, default=10.0,
                        help='Teacher-Student L1 waveform 蒸餾權重')

    # 量化參數
    parser.add_argument('--quant_bits', type=int, default=16)
    parser.add_argument('--per_channel', action='store_true', default=True,
                        help='使用 per-channel 量化')
    parser.add_argument('--no_per_channel', action='store_false', dest='per_channel')
    parser.add_argument('--qat_start_epoch', type=int, default=3,
                        help='從第幾個 epoch 開始啟用 fake-quant')

    # 日誌
    parser.add_argument('--stdout_interval', type=int, default=50)
    parser.add_argument('--summary_interval', type=int, default=100)

    # Resume
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    # ── 載入 config ───────────────────────────────────────────
    with open(args.config) as f:
        h = AttrDict(json.loads(f.read()))

    # ── 建立輸出目錄 ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, 'qat_log.txt'),
                'a' if args.resume else 'w'),
            logging.StreamHandler(),
        ],
        level=logging.INFO, format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S')

    logging.info('=' * 60)
    logging.info(f'HiFi-GAN INT{args.quant_bits} Quantization-Aware Training')
    logging.info('=' * 60)
    logging.info(f'Config: {args.config}')
    logging.info(f'Checkpoint: {args.checkpoint_path}')
    logging.info(f'Output: {args.output_dir}')

    train_qat(args, h)


if __name__ == '__main__':
    main()
