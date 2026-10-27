# -*- coding: utf-8 -*-
"""
finetune_custom_emg.py - 自行錄製 EMG 與對應語音微調 SNN-Mamba 模型腳本
支援兩種微調策略：
  1. 方案 A (--mode full):
     全模型微調 (端到端 CNN-SNN + Mamba + Readout)
    python finetune_custom_emg.py \
  --mode full \
  --output_dir finetune_output_scheme_a \
  --epochs 40 \
  --learning_rate 5e-5 \
  --emg_scale 50.0
  2. 方案 B (--mode head_only):
     凍結 Mamba 主幹 (Backbone)，僅微調前端 CNN 卷積塊與輸出 Linear Readout
    python finetune_custom_emg.py \
  --mode head_only \
  --output_dir finetune_output_scheme_b \
  --epochs 40 \
  --learning_rate 1e-4 \
  --emg_scale 50.0
"""

import os
import sys
import glob
import time
import pickle
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocoder import Vocoder
sys.modules.pop('utils', None)

from data_utils import load_audio, FeatureNormalizer
from architecture_spikelinear_noslim_selftune import Model
from align import align_from_distances

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('recordings_dir', 'recordings', '自錄資料夾 (包含 *_emg.npy 與 *_audio.wav)')
flags.DEFINE_string('model_path', 'models/spike_model/spike_model.pt', '預訓練權重路徑')
flags.DEFINE_string('output_dir', 'finetune_output_scheme_a', '微調模型與評估音訊儲存目錄')
flags.DEFINE_enum('mode', 'full', ['full', 'head_only'], '微調模式: full (方案A: 全微調) 或 head_only (方案B: 凍結Mamba)')
flags.DEFINE_integer('epochs', 40, '微調訓練輪數')
flags.DEFINE_float('learning_rate', 5e-5, '學習率 (full建議 5e-5, head_only建議 1e-4)')
flags.DEFINE_float('weight_decay', 1e-5, '權重衰減')
flags.DEFINE_float('emg_scale', 50.0, 'EMG 輸入縮放除數 (當 adaptive_norm=False 時生效)')
flags.DEFINE_float('spike_threshold', 0.4, 'SNN 脈衝閾值')
flags.DEFINE_integer('val_count', 3, '留作驗證集的檔案數量')
flags.DEFINE_bool('use_dtw', True, '是否使用 DTW 動態時間規整損失')
flags.DEFINE_integer('eval_audio_interval', 5, '每隔幾個 Epoch 匯出驗證集合成語音 .wav')

# --- 增強與清晰度優化參數 ---
flags.DEFINE_bool('augment', True, '訓練時是否啟用肌電資料增強 (通道增益擾動、微噪聲、隨機丟棄、時域遮蔽)')
flags.DEFINE_bool('adaptive_norm', True, '是否啟用各通道自適應標準化 (對齊官方 Gaddy 各通道能量比例，避免通道能量失衡)')
flags.DEFINE_float('silence_penalty', 0.2, '對停頓靜音段的殘留輸出給予額外壓制懲罰，消除背景白雜訊')

# 官方 Gaddy 資料集 8 通道基準標準差
OFFICIAL_STD = np.array([1.13, 3.01, 1.30, 2.37, 1.13, 1.68, 1.07, 0.73], dtype=np.float32)


def augment_emg(raw_emg):
    """
    肌電 4 重資料增強 (Data Augmentation):
    1. 隨機通道獨立增益 (0.85 ~ 1.15): 模擬貼片接觸阻抗自然變異
    2. 隨機微小高斯噪聲 (std=0.02): 破壞高頻死記硬背
    3. 隨機通道丟棄 (15% 機率): 防止過度依賴單一最強通道
    4. 隨機局部時域遮蔽 (25% 機率，遮蔽 20~35 點約 30~50ms): 強迫 Mamba 發揮記憶補足發音
    """
    raw_emg = raw_emg.copy()

    # 1. 隨機通道增益
    gains = np.random.uniform(0.85, 1.15, size=(1, 8)).astype(np.float32)
    raw_emg = raw_emg * gains

    # 2. 隨機高斯微噪聲
    noise = np.random.normal(0, 0.02, size=raw_emg.shape).astype(np.float32)
    raw_emg = raw_emg + noise

    # 3. 隨機通道丟棄
    if np.random.rand() < 0.15:
        drop_ch = np.random.randint(0, 8)
        raw_emg[:, drop_ch] = 0.0

    # 4. 局部時域遮蔽
    if np.random.rand() < 0.25:
        mask_len = np.random.randint(15, 35)
        if raw_emg.shape[0] > mask_len:
            t_start = np.random.randint(0, raw_emg.shape[0] - mask_len)
            raw_emg[t_start:t_start + mask_len, :] = 0.0

    return raw_emg


class CustomEMGDataset(torch.utils.data.Dataset):
    def __init__(self, file_pairs, emg_scale=50.0, mfcc_norm=None, is_train=False):
        self.file_pairs = file_pairs
        self.emg_scale = emg_scale
        self.mfcc_norm = mfcc_norm
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emg_path, audio_path = self.file_pairs[idx]
        base_name = os.path.basename(emg_path)[:-len('_emg.npy')]

        raw_emg = np.load(emg_path).astype(np.float32)

        # 1. 通道動態範圍平衡與正規化
        if FLAGS.adaptive_norm:
            mean = raw_emg.mean(axis=0, keepdims=True)
            std = raw_emg.std(axis=0, keepdims=True) + 1e-4
            raw_emg = ((raw_emg - mean) / std) * OFFICIAL_STD
            raw_emg = 50.0 * np.tanh(raw_emg / 50.0)
        else:
            raw_emg = raw_emg / self.emg_scale
            raw_emg = 50.0 * np.tanh(raw_emg / 50.0)

        # 2. 訓練集專屬肌電資料增強
        if self.is_train and FLAGS.augment:
            raw_emg = augment_emg(raw_emg)

        # 3. 讀取並前處理音訊 Mel 頻譜
        mel = load_audio(audio_path)
        if self.mfcc_norm is not None:
            mel = self.mfcc_norm.normalize(mel)

        # 4. 時間維度對齊 (8 EMG 採樣點 = 1 Mel 幀)
        n_frames = min(raw_emg.shape[0] // 8, mel.shape[0])
        raw_emg = raw_emg[:n_frames * 8, :]
        mel = mel[:n_frames, :]

        return {
            'base_name': base_name,
            'emg': torch.from_numpy(raw_emg),
            'mel': torch.from_numpy(mel),
            'n_frames': n_frames
        }


def compute_loss(pred_mel, target_mel, use_dtw=True, silence_penalty=0.0):
    """
    計算預測 Mel 與真實 Mel 頻譜的損失：
    - 改採 L1 距離 (p=1.0) 做 DTW 對齊與損失，保留尖銳的共振峰 (Formants)
    - 加入靜音段殘留壓制項，消除背景白雜訊
    """
    if use_dtw:
        pred_sq = pred_mel.squeeze(0)  # (T_pred, 80)
        tgt_sq = target_mel.squeeze(0) # (T_tgt, 80)

        dists = torch.cdist(pred_sq, tgt_sq, p=1.0)
        alignment = align_from_distances(dists.detach().cpu().numpy().T)

        aligned_pred = pred_sq[alignment, :]
        aligned_tgt = tgt_sq[:len(alignment), :]

        l1_loss = F.l1_loss(aligned_pred, aligned_tgt)

        if silence_penalty > 0.0:
            is_silent = (aligned_tgt.mean(dim=-1) < -0.4).float()
            silence_leakage = (aligned_pred.abs() * is_silent.unsqueeze(-1)).mean()
            return l1_loss + silence_penalty * silence_leakage
        else:
            return l1_loss
    else:
        min_len = min(pred_mel.shape[1], target_mel.shape[1])
        return F.l1_loss(pred_mel[:, :min_len, :], target_mel[:, :min_len, :])


def main(_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode_desc = '方案 A (全模型微調)' if FLAGS.mode == 'full' else '方案 B (凍結 Mamba，僅微調前端與輸出層)'
    aug_desc = '開啟 (4重肌電增強)' if FLAGS.augment else '關閉'
    norm_desc = '開啟 (對齊官方能量)' if FLAGS.adaptive_norm else '關閉'

    print('\n' + '=' * 65)
    print(f'[*] 啟動自錄 EMG 微調系統 (裝置: {device})')
    print(f'[*] 微調策略: {mode_desc}')
    print(f'[*] 儲存目錄: {FLAGS.output_dir}')
    print(f'[*] 資料增強 (Data Augmentation): {aug_desc}')
    print(f'[*] 自適應通道平衡 (Adaptive Norm): {norm_desc}')
    print(f'[*] 靜音底噪抑制 (Silence Penalty): {FLAGS.silence_penalty}')
    print('=' * 65 + '\n')

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    val_audio_dir = os.path.join(FLAGS.output_dir, 'val_audio')
    os.makedirs(val_audio_dir, exist_ok=True)

    emg_files = sorted(glob.glob(os.path.join(FLAGS.recordings_dir, '*_emg.npy')))
    paired_files = []
    for e in emg_files:
        base = e[:-len('_emg.npy')]
        wav = base + '_audio.wav'
        if os.path.exists(wav):
            paired_files.append((e, wav))

    if not paired_files:
        print(f'[!] 錯誤: 在 {FLAGS.recordings_dir} 找不到任何成對的 *_emg.npy 與 *_audio.wav！')
        return

    print(f'[+] 成功配對 {len(paired_files)} 筆資料。')

    val_count = min(FLAGS.val_count, max(1, len(paired_files) // 5))
    train_pairs = paired_files[:-val_count]
    val_pairs = paired_files[-val_count:]

    print(f'    - 訓練集 (Train): {len(train_pairs)} 筆 (已套用資料增強)')
    print(f'    - 驗證集 (Val)  : {len(val_pairs)} 筆 (純淨無增強，用於精準評估)')
    for p in val_pairs:
        print(f'      * 驗證檔案: {os.path.basename(p[0])}')

    print('\n[*] 正在載入 FeatureNormalizer 與 HiFi-GAN Vocoder...')
    with open(FLAGS.normalizers_file, 'rb') as f:
        mfcc_norm, _ = pickle.load(f)
    vocoder = Vocoder()

    train_dataset = CustomEMGDataset(train_pairs, emg_scale=FLAGS.emg_scale, mfcc_norm=mfcc_norm, is_train=True)
    val_dataset = CustomEMGDataset(val_pairs, emg_scale=FLAGS.emg_scale, mfcc_norm=mfcc_norm, is_train=False)

    print(f'[*] 正在從 {FLAGS.model_path} 載入預訓練權重...')
    state = torch.load(FLAGS.model_path, map_location='cpu', weights_only=True)
    num_speech_features = state['w_out.weight'].shape[0]
    n_phones = state['w_aux.weight'].shape[0] if 'w_aux.weight' in state else None

    model = Model(8, num_speech_features, n_phones, spike_threshold=FLAGS.spike_threshold).to(device)
    model.load_state_dict(state, strict=True)

    if FLAGS.mode == 'head_only':
        print('\n[模式設定 - 方案 B]')
        print('  >> 凍結中間 Mamba 骨幹層，僅開放前端 conv_blocks, w_raw_in, final_norm, w_out 學習。')
        for p in model.parameters():
            p.requires_grad = False
        for p in model.conv_blocks.parameters():
            p.requires_grad = True
        for p in model.w_raw_in.parameters():
            p.requires_grad = True
        for p in model.final_norm.parameters():
            p.requires_grad = True
        for p in model.w_out.parameters():
            p.requires_grad = True
    else:
        print('\n[模式設定 - 方案 A]')
        print('  >> 全模型開放訓練 (端到端 CNN + Mamba + Readout)。')
        for p in model.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f'  >> 可訓練參數: {num_trainable:,} / 總參數: {num_total:,} ({num_trainable / num_total:.1%})')

    optimizer = torch.optim.AdamW(trainable_params, lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.epochs, eta_min=1e-6)

    print('\n[*] 產生初始未微調 (Epoch 0) 之驗證集音訊以供對照...')
    model.eval()
    with torch.no_grad():
        for item in val_dataset:
            x = item['emg'].unsqueeze(0).to(device)
            sess = torch.zeros(1, item['n_frames'], dtype=torch.long, device=device)
            pred, _ = model(None, x, sess)
            y_inv = mfcc_norm.inverse(pred.squeeze(0).cpu()).to(device)
            audio = vocoder(y_inv).cpu().numpy()
            out_path = os.path.join(val_audio_dir, f"{item['base_name']}_epoch00_baseline.wav")
            sf.write(out_path, audio, 22050)
    print(f'[+] 初始對照音訊已存至 {val_audio_dir}/')

    print('\n' + '=' * 65)
    print('🚀 開始微調訓練 (Fine-Tuning Loop)')
    print('=' * 65)

    best_val_loss = float('inf')

    for epoch in range(1, FLAGS.epochs + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad()

        indices = np.random.permutation(len(train_dataset))
        for idx in indices:
            item = train_dataset[idx]
            x = item['emg'].unsqueeze(0).to(device)
            y_tgt = item['mel'].unsqueeze(0).to(device)
            sess = torch.zeros(1, item['n_frames'], dtype=torch.long, device=device)

            pred, _ = model(None, x, sess)
            loss = compute_loss(pred, y_tgt, use_dtw=FLAGS.use_dtw, silence_penalty=FLAGS.silence_penalty)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for item in val_dataset:
                x = item['emg'].unsqueeze(0).to(device)
                y_tgt = item['mel'].unsqueeze(0).to(device)
                sess = torch.zeros(1, item['n_frames'], dtype=torch.long, device=device)
                pred, _ = model(None, x, sess)
                v_loss = compute_loss(pred, y_tgt, use_dtw=FLAGS.use_dtw, silence_penalty=FLAGS.silence_penalty)
                val_losses.append(v_loss.item())

        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, 'model_best.pt'))

        should_export_audio = (epoch % FLAGS.eval_audio_interval == 0) or is_best or (epoch == FLAGS.epochs)
        if should_export_audio:
            with torch.no_grad():
                for item in val_dataset:
                    x = item['emg'].unsqueeze(0).to(device)
                    sess = torch.zeros(1, item['n_frames'], dtype=torch.long, device=device)
                    pred, _ = model(None, x, sess)
                    y_inv = mfcc_norm.inverse(pred.squeeze(0).cpu()).to(device)
                    audio = vocoder(y_inv).cpu().numpy()
                    out_path = os.path.join(val_audio_dir, f"{item['base_name']}_epoch{epoch:02d}.wav")
                    sf.write(out_path, audio, 22050)

        marker = '⭐ [BEST]' if is_best else ''
        audio_tag = ' (已匯出試聽音訊)' if should_export_audio else ''
        print(f'Epoch [{epoch:02d}/{FLAGS.epochs:02d}] | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {marker}{audio_tag}')

    torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, 'model_last.pt'))
    print('\n' + '=' * 65)
    print(f'🎉 微調完成！最佳驗證 Loss: {best_val_loss:.4f}')
    print(f'    - 最佳權重檔: {os.path.join(FLAGS.output_dir, "model_best.pt")}')
    print(f'    - 最終權重檔: {os.path.join(FLAGS.output_dir, "model_last.pt")}')
    print(f'    - 試聽音訊檔: {val_audio_dir}/')
    print('=' * 65 + '\n')


if __name__ == '__main__':
    app.run(main)
