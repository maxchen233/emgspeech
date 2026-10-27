# -*- coding: utf-8 -*-
"""
=============================================================================
clean_recordings_audio.py - 自錄語音批次降噪、濾波與音量自動放大工具
=============================================================================
功能說明：
  1. 自動備份：處理前先將原始音訊備份至 recordings/raw_audio_backup/ (100% 安全)
  2. 80Hz 高通濾波：消除呼吸噴麥 (Plosives)、低頻震動與直流偏移
  3. 頻譜降噪 (noisereduce)：採集開頭靜音樣本，精準消除房間環境音與電流底噪
  4. 峰值正規化與音量提升：將微弱的麥克風訊號自動放大 10~20 dB 至標準電平 (0.88)
  5. 支援一鍵還原 (--restore)：隨時可無損還原回最初始的錄音

用法：
  python clean_recordings_audio.py                 # 執行降噪與音量放大
  python clean_recordings_audio.py --restore       # 還原為備份的原始檔案
=============================================================================
"""

import os
import sys
import glob
import shutil
import argparse
import numpy as np
import soundfile as sf
import scipy.signal
import noisereduce as nr


def process_audio_file(file_path, target_peak=0.88, prop_decrease=0.85):
    data, sr = sf.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0] # 轉為單聲道

    orig_peak = float(np.abs(data).max())
    orig_rms = float(np.sqrt(np.mean(data**2)))

    if orig_peak < 1e-5:
        print(f"  [!] 警告: {os.path.basename(file_path)} 音訊全靜音，略過處理。")
        return False, orig_peak, orig_peak, orig_rms, orig_rms, 0.0

    # 1. 80Hz 三階 Butterworth 高通濾波 (消除噴麥氣流與低頻嗡鳴)
    b, a = scipy.signal.butter(3, 80.0 / (sr / 2.0), btype='highpass')
    filtered = scipy.signal.filtfilt(b, a, data)

    # 2. 頻譜降噪：採樣開頭約 0.35 秒作為環境底噪特徵
    noise_sample_len = min(int(sr * 0.35), len(filtered) // 5)
    noise_sample = filtered[:noise_sample_len]
    denoised = nr.reduce_noise(
        y=filtered,
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=prop_decrease,
        stationary=True
    )

    # 3. 峰值動態放大至 target_peak (-1.1 dBFS)
    current_peak = float(np.abs(denoised).max())
    if current_peak > 1e-4:
        gain = target_peak / current_peak
        boosted = denoised * gain
    else:
        gain = 1.0
        boosted = denoised

    boosted = np.clip(boosted, -1.0, 1.0).astype(np.float32)
    new_peak = float(np.abs(boosted).max())
    new_rms = float(np.sqrt(np.mean(boosted**2)))
    gain_db = 20.0 * np.log10(new_peak / (orig_peak + 1e-9))

    # 覆蓋寫入乾淨音訊 (16-bit PCM WAV)
    sf.write(file_path, boosted, sr, subtype='PCM_16')

    return True, orig_peak, new_peak, orig_rms, new_rms, gain_db


def main():
    parser = argparse.ArgumentParser(description="自錄語音批次降噪、高通濾波與音量自動放大工具")
    parser.add_argument('--dir', default='recordings', help='音訊目錄 (預設: recordings)')
    parser.add_argument('--target_peak', type=float, default=0.88, help='目標峰值振幅 (預設: 0.88, 約 -1.1 dBFS)')
    parser.add_argument('--noise_reduce', type=float, default=0.85, help='降噪強度 (0.0~1.0, 預設: 0.85)')
    parser.add_argument('--restore', action='store_true', help='從備份還原原始音訊')
    args = parser.parse_args()

    recordings_dir = args.dir
    backup_dir = os.path.join(recordings_dir, 'raw_audio_backup')

    if args.restore:
        if not os.path.exists(backup_dir):
            print(f"[!] 找不到備份目錄: {backup_dir}，無法還原。")
            return
        backup_files = glob.glob(os.path.join(backup_dir, '*_audio.wav'))
        print(f"\n[*] 正在從備份目錄還原 {len(backup_files)} 個原始音訊檔案...")
        for bf in backup_files:
            dest = os.path.join(recordings_dir, os.path.basename(bf))
            shutil.copy2(bf, dest)
        print(f"[+] 還原完成！已將原始錄音複製回 {recordings_dir}/")
        return

    wav_files = sorted(glob.glob(os.path.join(recordings_dir, '*_audio.wav')))
    if not wav_files:
        print(f"[!] 在 {recordings_dir} 找不到任何 *_audio.wav 檔案！")
        return

    print("\n" + "=" * 75)
    print("🎙️  自錄語音批次降噪、高通濾波與音量放大工具")
    print(f"[*] 音訊目錄: {recordings_dir}")
    print(f"[*] 檔案數量: {len(wav_files)} 筆")
    print(f"[*] 目標峰值: {args.target_peak:.2f} | 降噪強度: {args.noise_reduce:.2f}")
    print("=" * 75)

    # 1. 建立安全備份
    os.makedirs(backup_dir, exist_ok=True)
    print(f"\n[1/3] 正在安全備份原始音訊至: {backup_dir}/")
    for wf in wav_files:
        backup_path = os.path.join(backup_dir, os.path.basename(wf))
        if not os.path.exists(backup_path):
            shutil.copy2(wf, backup_path)
    print(f"  [+] 備份完畢，任何時候皆可使用 --restore 一鍵還原。")

    # 2. 批次處理
    print(f"\n[2/3] 正在執行 80Hz 高通濾波 + 頻譜降噪 + 音量增益正規化...")
    print("-" * 75)
    print(f"{'檔名':<36} | {'原Peak':<7} -> {'新Peak':<7} | {'原RMS':<7} -> {'新RMS':<7} | {'增益 (dB)':<8}")
    print("-" * 75)

    total_boost = []
    for wf in wav_files:
        fname = os.path.basename(wf)
        ok, op, npk, orms, nrms, gdb = process_audio_file(wf, target_peak=args.target_peak, prop_decrease=args.noise_reduce)
        if ok:
            total_boost.append(gdb)
            print(f"{fname:<36} | {op:7.4f} -> {npk:7.4f} | {orms:7.4f} -> {nrms:7.4f} | +{gdb:5.2f} dB")

    avg_boost = np.mean(total_boost) if total_boost else 0.0
    print("-" * 75)
    print(f"\n[3/3] 🎉 全部處理完成！")
    print(f"  - 平均音量提升: +{avg_boost:.2f} dB (徹底解決音量過小、訊噪比低下的問題)")
    print(f"  - 80Hz 以下噴麥風切聲已消除")
    print(f"  - 靜態背景電流與環境雜訊已消除")
    print(f"  - 處理後音訊已直接更新回: {recordings_dir}/")
    print(f"  - 備份檔案保存在: {backup_dir}/ (隨時可還原)")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    main()
