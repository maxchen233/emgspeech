"""
Mel Spectrogram → 語音轉換工具

從 .npy 檔案讀取 mel spectrogram，透過 HiFi-GAN vocoder 合成語音。

用法:
    # 基本用法 (mel 已經是 denormalized 的原始值)
    python mel_to_audio.py --mel_path output/mel.npy

    # mel 是模型輸出 (需要 inverse normalize)
    python mel_to_audio.py --mel_path output/mel.npy --denormalize

    # 指定輸出路徑
    python mel_to_audio.py --mel_path output/mel.npy --output output/synthesized.wav

    # 使用 streaming vocoder (分段合成，避免邊界 artifact)
    python mel_to_audio.py --mel_path output/mel.npy --streaming --vocoder_chunk_size 30

    # 指定 vocoder checkpoint
    python mel_to_audio.py --mel_path output/mel.npy \
        --hifigan_checkpoint hifigan_finetuned/checkpoint

mel.npy 格式:
    - shape = (T, 80): T 幀, 80 個 mel bin
    - dtype: float32
"""

import os
import sys
import pickle
import logging
import time

import numpy as np
import torch
import soundfile as sf

from vocoder import Vocoder, StreamingVocoder

from absl import flags
FLAGS = flags.FLAGS

# ── 參數定義 ─────────────────────────────────────────────────
flags.DEFINE_string('mel_path', None,
                    'Mel spectrogram .npy 檔案路徑 (shape=(T, 80))')
flags.DEFINE_string('output', None,
                    '輸出 WAV 路徑 (預設: 與 mel 同目錄, 副檔名改 .wav)')
flags.DEFINE_boolean('denormalize', False,
                     '是否對 mel 做 inverse normalize '
                     '(適用於模型直接輸出的 normalized mel)')
flags.DEFINE_boolean('streaming', False,
                     '使用 StreamingVocoder 分段合成 (避免邊界 artifact)')
flags.DEFINE_integer('vocoder_chunk_size', 30,
                     'StreamingVocoder 每次處理的 mel 幀數')

AUDIO_SR = 22050


def load_mel(mel_path):
    """載入 mel spectrogram .npy 檔案"""
    logging.info(f'載入 mel: {mel_path}')
    mel = np.load(mel_path)

    if mel.ndim == 1:
        raise ValueError(f'mel.npy 應為 2D (T, 80), 但得到 1D shape={mel.shape}')

    # 支援 (80, T) 格式 → 自動轉置為 (T, 80)
    if mel.ndim == 2 and mel.shape[0] == 80 and mel.shape[1] != 80:
        logging.info(f'  偵測到 (80, T) 格式, 自動轉置為 (T, 80)')
        mel = mel.T

    if mel.ndim == 3:
        # 支援 (1, 80, T) 批次格式
        if mel.shape[0] == 1 and mel.shape[1] == 80:
            logging.info(f'  偵測到 (1, 80, T) 批次格式, 轉為 (T, 80)')
            mel = mel.squeeze(0).T
        else:
            raise ValueError(f'不支援的 3D mel shape: {mel.shape}')

    assert mel.shape[1] == 80, \
        f'mel 特徵維度應為 80, 但得到 {mel.shape[1]} (shape={mel.shape})'

    logging.info(f'  mel shape: {mel.shape} ({mel.shape[0]} 幀, '
                 f'約 {mel.shape[0] * 256 / AUDIO_SR:.2f}s)')
    logging.info(f'  值域: [{mel.min():.4f}, {mel.max():.4f}]')

    return mel.astype(np.float32)


def denormalize_mel(mel):
    """使用 normalizers.pkl 對 mel 做 inverse normalization"""
    norm_file = FLAGS.normalizers_file
    logging.info(f'載入 normalizer: {norm_file}')

    with open(norm_file, 'rb') as f:
        mfcc_norm, _emg_norm = pickle.load(f)

    mel_tensor = torch.tensor(mel, dtype=torch.float32)
    mel_denorm = mfcc_norm.inverse(mel_tensor)
    logging.info(f'  inverse normalize 完成, '
                 f'值域: [{mel_denorm.min():.4f}, {mel_denorm.max():.4f}]')
    return mel_denorm


def synthesize_batch(vocoder, mel_tensor, device):
    """批次 vocoder 合成 (一次送入整段 mel)"""
    logging.info('使用批次 vocoder 合成...')
    mel_tensor = mel_tensor.to(device)
    audio = vocoder(mel_tensor).cpu().numpy()
    return audio


def synthesize_streaming(vocoder, mel_tensor, device, chunk_size):
    """StreamingVocoder 分段合成"""
    logging.info(f'使用 StreamingVocoder 分段合成 (chunk_size={chunk_size})...')
    streaming_vocoder = StreamingVocoder(vocoder)
    mel_tensor = mel_tensor.to(device)

    total_frames = mel_tensor.shape[0]
    audio_chunks = []
    n_chunks = 0

    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunk = mel_tensor[start:end]
        audio = streaming_vocoder.synthesize_chunk(chunk)
        if len(audio) > 0:
            audio_chunks.append(audio)
            n_chunks += 1

    # flush 最後一段
    tail = streaming_vocoder.flush()
    if len(tail) > 0:
        audio_chunks.append(tail)
        n_chunks += 1

    logging.info(f'  共合成 {n_chunks} 個 chunk')
    return np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)


def main():
    if FLAGS.mel_path is None:
        print('錯誤: 請指定 --mel_path')
        print('用法: python mel_to_audio.py --mel_path <mel.npy>')
        sys.exit(1)

    # ── 設定 logging ──────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler()]
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Device: {device}')

    # ── 載入 mel ──────────────────────────────────────────────
    mel = load_mel(FLAGS.mel_path)

    # ── Inverse normalize (可選) ──────────────────────────────
    if FLAGS.denormalize:
        mel_tensor = denormalize_mel(mel)
    else:
        mel_tensor = torch.tensor(mel, dtype=torch.float32)

    # ── 載入 Vocoder ──────────────────────────────────────────
    logging.info('載入 HiFi-GAN vocoder...')
    start_time = time.time()
    vocoder = Vocoder(device=device)
    logging.info(f'  vocoder 載入完成 ({time.time() - start_time:.2f}s)')

    # ── 合成語音 ──────────────────────────────────────────────
    start_time = time.time()
    if FLAGS.streaming:
        audio = synthesize_streaming(
            vocoder, mel_tensor, device, FLAGS.vocoder_chunk_size
        )
    else:
        audio = synthesize_batch(vocoder, mel_tensor, device)
    synth_time = time.time() - start_time

    if len(audio) == 0:
        logging.error('合成失敗: 未產生任何音訊')
        sys.exit(1)

    # ── 儲存 WAV ──────────────────────────────────────────────
    if FLAGS.output:
        output_path = FLAGS.output
    else:
        # 與 mel 同目錄, 副檔名改為 .wav
        output_path = os.path.splitext(FLAGS.mel_path)[0] + '.wav'

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, audio, AUDIO_SR)

    # ── 統計 ──────────────────────────────────────────────────
    audio_dur = len(audio) / AUDIO_SR
    logging.info('=' * 50)
    logging.info(f'  ✅ 合成完成')
    logging.info(f'  輸入 mel:    {mel.shape[0]} 幀')
    logging.info(f'  輸出音訊:    {audio_dur:.2f}s ({len(audio)} samples)')
    logging.info(f'  合成耗時:    {synth_time:.3f}s')
    logging.info(f'  RTF:         {synth_time / audio_dur:.4f}x')
    logging.info(f'  輸出檔案:    {output_path}')
    logging.info('=' * 50)


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
