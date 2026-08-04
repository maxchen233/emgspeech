"""
即時 EMG → 語音串流推論引擎 (多執行緒 Pipeline 版)

用法:
    python realtime_inference.py \
        --hifigan_checkpoint hifigan_finetuned/checkpoint \
        --model_path models/transduction_model/model.pt \
        --sample_index 0 \
        --streaming_output output/streaming_output.wav

    # 輸出每段 vocoder chunk:
    python realtime_inference.py \
        --hifigan_checkpoint hifigan_finetuned/checkpoint \
        --model_path models/transduction_model/model.pt \
        --sample_index 0 \
        --streaming_output output/streaming_output.wav \
        --save_chunks --noplay_audio

架構:
    Thread 1: EMG 擷取    → emg_queue
    Thread 2: 前處理       → processed_queue
    Thread 3: 推論引擎     → audio_segments[]
        因果 CNN (conv cache) → Stateful Mamba (SSM state) → Vocoder (trim)
    播放: paplay (推論完成後)
"""

import os
import sys
import time
import numpy as np
import logging

import torch
import soundfile as sf

from architecture import Model
from data_utils import phoneme_inventory
from vocoder import Vocoder, StreamingVocoder
from emg_simulator import EMGSimulator
from read_emg import EMGDataset
from streaming_pipeline import StreamingPipeline, AUDIO_SR
from quantize_int16 import load_quantized_model

from absl import flags
FLAGS = flags.FLAGS

# ── 串流推論專用參數 ─────────────────────────────────────────
flags.DEFINE_integer('vocoder_chunk_size', 30,
                     '每次送入 vocoder 的 mel 幀數 (越大越穩定，但延遲越高)')
flags.DEFINE_integer('emg_chunk_frames', 10,
                     '每個 EMG chunk 對應的 mel 幀數')
flags.DEFINE_integer('sample_index', 0,
                     '使用資料集中第幾個樣本')
flags.DEFINE_string('emg_file', None,
                    '自訂 EMG 檔案路徑 (.npy, shape=(T,8)). 設定後忽略 sample_index')
flags.DEFINE_boolean('list_samples', False,
                     '列出資料集中所有可用樣本後結束')
try:
    flags.DEFINE_string('model_path',
                        'models/transduction_model/model.pt',
                        '模型檔案路徑')
except flags.DuplicateFlagError:
    pass
flags.DEFINE_string('streaming_output',
                    'output/streaming_output.wav',
                    '串流輸出 WAV 路徑')
flags.DEFINE_boolean('realtime', False,
                     '是否模擬真實取樣速率延遲')
flags.DEFINE_boolean('compare_batch', True,
                     '是否同時執行批次推論做比較')
flags.DEFINE_string('play_mode', 'paplay',
                    '播放模式: realtime (即時 pacat), paplay (推論完再播), none (不播)')
flags.DEFINE_boolean('save_wav', True,
                     '是否儲存 WAV 檔案')
flags.DEFINE_boolean('save_chunks', False,
                     '是否儲存每段 vocoder 輸出為個別 WAV')

try:
    flags.DEFINE_boolean('quantized', False, '是否使用 INT16 量化模型')
except flags.DuplicateFlagError:
    pass


def run_batch_inference(model, datapoint, device, audio_normalizer, vocoder):
    """批次推論 (用於比較)"""
    model.eval()
    with torch.no_grad():
        sess = datapoint['session_ids'].to(device).unsqueeze(0)
        X = datapoint['emg'].to(torch.float32, device=device).unsqueeze(0)
        X_raw = datapoint['raw_emg'].to(torch.float32, device=device).unsqueeze(0)
        pred, _ = model(X, X_raw, sess)
        y = pred.squeeze(0)
        y = audio_normalizer.inverse(y.cpu()).to(device)
        audio = vocoder(y).cpu().numpy()
    return audio


def main():
    out_dir = os.path.dirname(FLAGS.streaming_output) or 'output'
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(out_dir, 'streaming_log.txt'), 'w'),
            logging.StreamHandler()
        ],
        level=logging.INFO,
        format="%(asctime)s %(message)s"
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Device: {device}')

    # ── 載入資料集 ────────────────────────────────────────────
    logging.info('載入資料集...')
    devset = EMGDataset(dev=True)

    # 列出所有樣本
    if FLAGS.list_samples:
        print(f'\n資料集共 {len(devset)} 個樣本:')
        print(f'{"idx":>4s}  {"raw_emg shape":>16s}  {"mel frames":>10s}  {"時長(s)":>8s}')
        print('-' * 50)
        for i in range(len(devset)):
            s = devset[i]
            raw = s['raw_emg'].numpy()
            dur = len(raw) / 689.06
            print(f'{i:4d}  {str(raw.shape):>16s}  {len(raw)//8:>10d}  {dur:>8.2f}')
        return

    # 載入 EMG 資料
    if FLAGS.emg_file:
        logging.info(f'載入自訂 EMG: {FLAGS.emg_file}')
        # 載入自訂 EMG 並套用與資料集相同的前處理流程
        raw_emg = np.load(FLAGS.emg_file)
        if raw_emg.ndim == 1:
            raw_emg = raw_emg.reshape(-1, 8)
        assert raw_emg.shape[1] == 8, f'EMG 應為 (T, 8), 得到 {raw_emg.shape}'
        # 前處理：與 load_utterance + __getitem__ 完全一致
        from read_emg import apply_to_all, notch_harmonics, remove_drift, subsample
        raw_emg = apply_to_all(notch_harmonics, raw_emg, 60, 1000)
        raw_emg = apply_to_all(remove_drift, raw_emg, 1000)
        # raw_emg = emg_orig，只 subsample 到 689.06 Hz（不做 516.79）
        raw_emg = apply_to_all(subsample, raw_emg, 689.06, 1000)
        for c in FLAGS.remove_channels:
            raw_emg[:, int(c)] = 0
        # 與 __getitem__ 相同的正規化
        raw_emg = raw_emg.astype(np.float32)
        raw_emg = raw_emg / 20
        raw_emg = 50 * np.tanh(raw_emg / 50.)
        sample = None
        logging.info(f'自訂 EMG: shape={raw_emg.shape}, '
                     f'總幀數={len(raw_emg)//8}, '
                     f'時長={len(raw_emg)/689.06:.2f}s')
    else:
        sample = devset[FLAGS.sample_index]
        raw_emg = sample['raw_emg'].numpy()
        logging.info(f'樣本 {FLAGS.sample_index}: raw_emg shape={raw_emg.shape}, '
                     f'總幀數={len(raw_emg)//8}')

    # ── 載入模型 ──────────────────────────────────────────────
    logging.info(f'載入模型: {FLAGS.model_path}')
    n_phones = len(phoneme_inventory)
    model = Model(devset.num_features, devset.num_speech_features,
                  n_phones).to(device)
    if FLAGS.quantized:
        load_quantized_model(model, FLAGS.model_path, device)
    else:
        state_dict = torch.load(FLAGS.model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()

    # ── 載入 Vocoder ──────────────────────────────────────────
    logging.info('載入 HiFi-GAN vocoder...')
    vocoder = Vocoder(device=device)
    streaming_vocoder = StreamingVocoder(vocoder)

    # ── 建立 EMG 模擬器 ──────────────────────────────────────
    simulator = EMGSimulator(
        raw_emg=raw_emg,
        chunk_frames=FLAGS.emg_chunk_frames,
        realtime=FLAGS.realtime
    )
    logging.info(f'EMG 模擬器: {simulator.total_chunks} chunks, '
                 f'每 chunk {FLAGS.emg_chunk_frames} 幀, '
                 f'信號時長 {simulator.duration_seconds:.2f}s')

    # ── 建立多執行緒 Pipeline ─────────────────────────────────
    chunk_dir = None
    if FLAGS.save_chunks:
        chunk_dir = os.path.join(out_dir, 'chunks')
        logging.info(f' 每段音訊將存到: {chunk_dir}/')

    pipeline = StreamingPipeline(
        model=model,
        streaming_vocoder=streaming_vocoder,
        audio_normalizer=devset.mfcc_norm,
        device=device,
        vocoder_chunk_size=FLAGS.vocoder_chunk_size,
        play_mode=FLAGS.play_mode,
        chunk_output_dir=chunk_dir,
    )

    # ── 啟動 Pipeline ─────────────────────────────────────


    start_time = time.time()

    pipeline.start(emg_source=simulator)

    # wait() 會在推論完後依 play_mode 決定是否播放
    wav_for_play = FLAGS.streaming_output if FLAGS.play_mode == 'paplay' else None
    pipeline.wait(wav_path=wav_for_play)

    # ── 結果統計 ──────────────────────────────────────────────
    full_audio = pipeline.get_full_audio()
    if len(full_audio) > 0:
        audio_dur = len(full_audio) / AUDIO_SR
        total_time = time.time() - start_time

        if pipeline.inference_done_time:
            infer_time = pipeline.inference_done_time - start_time
        else:
            infer_time = total_time


        if FLAGS.save_wav:
            sf.write(FLAGS.streaming_output, full_audio, AUDIO_SR)
        logging.info('=' * 60)
    else:
        logging.warning(' 未產生任何音訊')
        return

    # ── (可選) 批次推論比較 ───────────────────────────────────
    if FLAGS.compare_batch and sample is None:
        logging.info('執行批次推論進行比較...')
        batch_audio = run_batch_inference(
            model, sample, device, devset.mfcc_norm, vocoder
        )
        batch_path = FLAGS.streaming_output.replace('.wav', '_batch.wav')
        sf.write(batch_path, batch_audio, AUDIO_SR)

        min_len = min(len(full_audio), len(batch_audio))
        if min_len > 0:
            diff = np.abs(full_audio[:min_len] - batch_audio[:min_len])
            logging.info(f' 批次 vs 串流比較:')
            logging.info(f'   串流音訊長度: {len(full_audio)} samples')
            logging.info(f'   批次音訊長度: {len(batch_audio)} samples')
            logging.info(f'   平均絕對差異: {diff.mean():.6f}')
            logging.info(f'   最大絕對差異: {diff.max():.6f}')
            logging.info(f'   批次輸出已儲存: {batch_path}')


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
