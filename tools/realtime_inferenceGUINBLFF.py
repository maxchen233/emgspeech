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
import glob
import queue
import threading
import numpy as np
import logging

import torch
import soundfile as sf
from pathlib import Path

# 將 hifi_gan 資料夾加入 Python 模組搜尋路徑
hifi_gan_path = Path(__file__).resolve().parent / "hifi_gan"
if str(hifi_gan_path) not in sys.path:
    sys.path.insert(0, str(hifi_gan_path))

from architecture import Model
from data_utils import phoneme_inventory
from vocoder import Vocoder, StreamingVocoder
from emg_simulator import EMGSimulator
from read_emg import EMGDataset
from streaming_pipeline import StreamingPipeline, AUDIO_SR
from cpp_streaming_pipeline import CppStreamingPipeline

try:
    from channel_quality import ChannelQualityAnalyzer, ChannelStatus
    HAS_CHANNEL_QUALITY = True
except ImportError:
    HAS_CHANNEL_QUALITY = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
        QPushButton, QLabel, QProgressBar, QFrame
    )
    from PySide6.QtCore import QThread, Signal, Slot, Qt, QRect
    from PySide6.QtGui import QPainter, QColor, QFont, QPen
    import pyqtgraph as pg
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

from absl import flags
FLAGS = flags.FLAGS

# ── 串流推論專用參數 ─────────────────────────────────────────
flags.DEFINE_boolean('gui', True,
                     '是否啟動 PySide6 GUI 視覺化介面 (預設開啟)')
flags.DEFINE_integer('vocoder_chunk_size', 50,
                     '每次送入 vocoder 的 mel 幀數 (越大越穩定，但延遲越高)')
flags.DEFINE_integer('emg_chunk_frames', 10,
                     '每個 EMG chunk 對應的 mel 幀數')
flags.DEFINE_integer('sample_index', 0,
                     '使用資料集中第幾個樣本')
flags.DEFINE_string('emg_file', None,
                    '自訂 EMG 檔案路徑 (.npy, shape=(T,8)). 設定後忽略 sample_index')
flags.DEFINE_boolean('list_samples', False,
                     '列出資料集中所有可用樣本後結束')
flags.DEFINE_string('model_path',
                    'models_mamba1/transduction_model/model.pt',
                    '模型檔案路徑')
flags.DEFINE_string('streaming_output',
                    'output/streaming_output.wav',
                    '串流輸出 WAV 路徑')
flags.DEFINE_boolean('realtime', False,
                     '是否模擬真實取樣速率延遲')
flags.DEFINE_boolean('compare_batch', True,
                     '是否同時執行批次推論做比較')
flags.DEFINE_string('play_mode', 'realtime',
                    '播放模式: realtime (即時 pacat), paplay (推論完再播), none (不播)')
flags.DEFINE_boolean('save_wav', True,
                     '是否儲存 WAV 檔案')
flags.DEFINE_boolean('save_chunks', False,
                     '是否儲存每段 vocoder 輸出為個別 WAV')
flags.DEFINE_boolean('use_cpp', False,
                     '使用 C++ 推論引擎 (不用 PyTorch model)')
flags.DEFINE_string('weights_dir', 'exported_weights',
                    'C++ 引擎的匯出權重目錄')
flags.DEFINE_boolean('use_int8', False,
                     '使用 INT8 量化模型 (qat_model_int8.pt)')
flags.DEFINE_string('int8_model_path',
                    'output_qat_int8/qat_model_int8.pt',
                    'INT8 量化模型檔案路徑')


# 8-channel anatomical muscle position labels (David Gaddy EMNLP 2020 / EECS-2022-68, Table 3)
CH_LABELS = [
    'Ch 1: Left Cheek (Risorius / Zygomaticus Major)',
    'Ch 2: Left Jaw Angle (Depressor Anguli Oris)',
    'Ch 3: Submental 3cm (Suprahyoid / Tongue Base)',
    'Ch 4: Thyroid Left 3cm (Thyrohyoid / Vocal Cords)',
    'Ch 5: Right Mandible (Right Masseter)',
    'Ch 6: Right Lower Cheek (Orbicularis Oris / Depressor)',
    'Ch 7: Right Nasolabial (Levator Labii Superioris)',
    'Ch 8: Right Preauricular (Posterior Masseter)'
]


# ── PySide6 GUI 模組 ──────────────────────────────────────────
if HAS_PYSIDE6:
    class SignalBarsWidget(QWidget):
        """手機信號強度風格的 EMG 通道品質指示器 (4 根由小到大的長條)"""
        COLOR_4BARS = QColor(76, 175, 80)       # 亮綠色 — 優秀
        COLOR_3BARS = QColor(139, 195, 74)      # 淺綠色 — 良好
        COLOR_2BARS = QColor(255, 193, 7)       # 琥珀黃 — 普通
        COLOR_1BAR  = QColor(255, 87, 34)       # 深橘紅 — 差
        COLOR_0BARS = QColor(244, 67, 54)       # 紅色   — 極差
        COLOR_GRAY  = QColor(100, 100, 100)     # 灰色   — 未校準
        COLOR_INACTIVE = QColor(50, 50, 50)     # 暗底
        COLOR_BORDER   = QColor(70, 70, 70)     # 邊框

        _BAR_COLORS = {4: COLOR_4BARS, 3: COLOR_3BARS, 2: COLOR_2BARS,
                       1: COLOR_1BAR,  0: COLOR_0BARS}

        def __init__(self, channel_id, parent=None):
            super().__init__(parent)
            self.channel_id = channel_id
            self._bars = 0
            self._calibrated = False
            self.setToolTip('Not calibrated')
            self.setFixedSize(58, 58)

        def set_status(self, status):
            if hasattr(status, 'quality') and status.quality == 'uncalibrated':
                self._bars = 0
                self._calibrated = False
            else:
                self._bars = getattr(status, 'bars', 0)
                self._calibrated = True
            self.setToolTip(getattr(status, 'detail', ''))
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            label_h = 14
            pad = 3
            bar_h_max = h - label_h - pad * 2
            bar_area_w = w - pad * 2
            n = 4
            gap = 3
            bar_w = max(6, (bar_area_w - (n - 1) * gap) // n)
            total_w = n * bar_w + (n - 1) * gap
            x0 = pad + (bar_area_w - total_w) // 2

            color = (self._BAR_COLORS.get(self._bars, self.COLOR_0BARS)
                     if self._calibrated else self.COLOR_GRAY)

            for i in range(n):
                frac = (i + 1) / n
                bh = int(bar_h_max * frac)
                x = x0 + i * (bar_w + gap)
                y = pad + bar_h_max - bh

                if not self._calibrated:
                    painter.setPen(QPen(self.COLOR_GRAY, 1))
                    painter.setBrush(QColor(40, 40, 40))
                elif i < self._bars:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(color)
                else:
                    painter.setPen(QPen(self.COLOR_BORDER, 1))
                    painter.setBrush(self.COLOR_INACTIVE)

                painter.drawRoundedRect(x, y, bar_w, bh, 2, 2)

            painter.setPen(QColor(180, 180, 180))
            font = painter.font()
            font.setPointSize(8)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QRect(0, h - label_h, w, label_h),
                             Qt.AlignCenter, f'Ch{self.channel_id + 1}')
            painter.end()

    class FilePlaybackWorker(QThread):
        update_plot_signal = Signal(np.ndarray)
        file_changed_signal = Signal(str, int, int)
        status_signal = Signal(str)
        ground_truth_audio_signal = Signal(np.ndarray)

        def __init__(self, search_root="emg_data", data_queue=None, crossfade_ms=200):
            super().__init__()
            self.search_root = FLAGS.emg_file or search_root
            self.data_queue = data_queue
            self.fps_delay = 50 
            self.ui_batch_size = 34
            self.inference_batch_size = 138 
            self.crossfade_samples = int(689.06 * crossfade_ms / 1000)
            
            self.is_running = True
            self.is_recording = False  
            self.inference_buffer = []
            
            self.file_list = []
            self.file_idx = 0
            self.data = None
            self.current_idx = 0
            self.data_length = 0

            # Ground-truth 音訊 (已 resample 至播放用 22050 Hz)
            self.gt_audio = None
            self.gt_audio_idx = 0
            self.gt_audio_length = 0
            self.playback_sr = 22050   # 與 sounddevice OutputStream 一致

            # 各通道動態歸零偏移量 (透過校準按鈕更新，完全對齊 draw_EMG)
            self.channel_offsets = np.zeros((8, 1), dtype=np.float32)

        def set_channel_offsets(self, offsets):
            self.channel_offsets = np.array(offsets, dtype=np.float32).reshape(8, 1)

        def _scan_emg_files(self, root_dir):
            if os.path.isfile(root_dir):
                return [root_dir]
            matches = []
            if os.path.exists(root_dir):
                for dirpath, dirnames, filenames in os.walk(root_dir):
                    for fname in sorted(filenames):
                        if fname.endswith('.npy') and 'emg' in fname.lower():
                            matches.append(os.path.join(dirpath, fname))
            matches.sort()
            return matches

        def _find_matching_audio(self, emg_path):
            """根據 EMG 檔名找同編號的 audio 檔案。
            例如: 101_emg.npy → 101_audio_clean.flac
            """
            import re
            dirpath = os.path.dirname(emg_path)
            basename = os.path.basename(emg_path)
            match = re.match(r'^(\d+)_emg', basename)
            if not match:
                return None
            num = match.group(1)
            for suffix in ['_audio_clean.flac', '_audio.flac',
                           '_audio_clean.wav', '_audio.wav']:
                full_path = os.path.join(dirpath, num + suffix)
                if os.path.exists(full_path):
                    return full_path
            return None

        def _load_ground_truth_audio(self, audio_path):
            """載入 ground-truth 音訊，resample 至 playback_sr (22050 Hz)。"""
            try:
                audio_data, file_sr = sf.read(audio_path, dtype='float32')
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                # Resample 至播放用 sample rate (22050 Hz)
                if file_sr != self.playback_sr:
                    from scipy.signal import resample_poly
                    from math import gcd
                    up = self.playback_sr
                    down = file_sr
                    g = gcd(up, down)
                    audio_data = resample_poly(audio_data, up // g, down // g).astype(np.float32)
                    logging.info(f"[Audio] Resample {file_sr} -> {self.playback_sr} Hz")
                # ── Distortion modeling ──
                from scipy.signal import butter, sosfilt
                # 1) Lowpass filter: cutoff above 6kHz
                sos = butter(4, 6000, btype='low', fs=self.playback_sr, output='sos')
                audio_data = sosfilt(sos, audio_data).astype(np.float32)
                # 2) Subtle noise
                noise = np.random.randn(len(audio_data)).astype(np.float32) * 0
                audio_data = audio_data + noise
                # 3) Soft clipping (tanh)
                audio_data = np.tanh(audio_data * 1.2).astype(np.float32)
                self.gt_audio = audio_data
                self.gt_audio_idx = 0
                self.gt_audio_length = len(audio_data)
                logging.info(f"[Audio] Loaded ground-truth: {os.path.basename(audio_path)} "
                           f"(orig sr={file_sr}, play sr={self.playback_sr}, "
                           f"samples={self.gt_audio_length}, "
                           f"duration={self.gt_audio_length/self.playback_sr:.2f}s)")
                return True
            except Exception as e:
                logging.warning(f"[Audio] Failed to load ground-truth: {audio_path}: {e}")
                self.gt_audio = None
                self.gt_audio_length = 0
                return False

        def _load_and_preprocess_data(self, path):
            if path.endswith('.npy'):
                x = np.load(path)
                if x.ndim == 1:
                    x = x.reshape(-1, 8)
                if x.shape[0] == 8 and x.shape[1] > 8:
                    x = x.T
            else:
                x = np.loadtxt(path, delimiter=',', skiprows=1)
                
            from read_emg import apply_to_all, notch_harmonics, remove_drift, subsample
            x = apply_to_all(notch_harmonics, x, 60, 1000)
            x = apply_to_all(remove_drift, x, 1000)
            x = apply_to_all(subsample, x, 689.06, 1000)
            x = x / 20.0
            x = 50.0 * np.tanh(x / 50.0)
            return x.T

        def _load_current_file(self):
            if not self.file_list:
                return
            path = self.file_list[self.file_idx]
            self.data = self._load_and_preprocess_data(path)
            self.current_idx = 0
            self.data_length = self.data.shape[1]
            logging.info(f"[Playback] Playing file: {path}  [{self.file_idx + 1}/{len(self.file_list)}]")
            # Load corresponding ground-truth audio
            audio_path = self._find_matching_audio(path)
            if audio_path:
                self._load_ground_truth_audio(audio_path)
            else:
                logging.warning(f"[Audio] Matching audio file not found: {path}")
                self.gt_audio = None
                self.gt_audio_length = 0
            self.file_changed_signal.emit(
                os.path.basename(path), self.file_idx + 1, len(self.file_list))

        def _apply_crossfade(self, outgoing_tail, incoming_head):
            n = min(outgoing_tail.shape[1], incoming_head.shape[1])
            if n == 0:
                return incoming_head
            fade_out = np.linspace(1.0, 0.0, n).reshape(1, -1)
            fade_in  = np.linspace(0.0, 1.0, n).reshape(1, -1)
            blended = outgoing_tail[:, :n] * fade_out + incoming_head[:, :n] * fade_in
            return blended

        def _advance_to_next_file(self):
            if len(self.file_list) <= 1:
                self.current_idx = 0
                return None
            cf_len = min(self.crossfade_samples, self.data_length)
            outgoing_tail = self.data[:, -cf_len:] if cf_len > 0 else None
            self.file_idx = (self.file_idx + 1) % len(self.file_list)
            self._load_current_file()
            if outgoing_tail is not None and self.data_length > 0:
                cf_len = min(cf_len, self.data_length)
                incoming_head = self.data[:, :cf_len]
                crossfaded = self._apply_crossfade(outgoing_tail, incoming_head)
                self.current_idx = cf_len
                return crossfaded
            return None

        def run(self):
            self.status_signal.emit("Status: Scanning EMG files...")
            self.file_list = self._scan_emg_files(self.search_root)
            if self.file_list:
                self._load_current_file()
                self.status_signal.emit(f"Status: Loaded {len(self.file_list)} files.")
            else:
                self.status_signal.emit("Status: No EMG files found, using standby noise.")

            self.was_recording = False
            crossfade_remainder = None
            cf_offset = 0

            while self.is_running:
                if self.is_recording:
                    if not self.file_list or self.data_length == 0:
                        batch_data = np.zeros((8, self.ui_batch_size))
                    elif crossfade_remainder is not None:
                        cf_remaining = crossfade_remainder.shape[1] - cf_offset
                        n = min(self.ui_batch_size, cf_remaining)
                        batch_data = crossfade_remainder[:, cf_offset:cf_offset + n]
                        cf_offset += n
                        if cf_offset >= crossfade_remainder.shape[1]:
                            crossfade_remainder = None
                            cf_offset = 0
                    else:
                        end_idx = self.current_idx + self.ui_batch_size
                        if end_idx <= self.data_length:
                            batch_data = self.data[:, self.current_idx:end_idx]
                            self.current_idx = end_idx
                        else:
                            remaining = self.data[:, self.current_idx:]
                            cf_data = self._advance_to_next_file()
                            if cf_data is not None:
                                crossfade_remainder = cf_data
                                cf_offset = 0
                            batch_data = remaining if remaining.shape[1] > 0 else np.zeros((8, self.ui_batch_size))

                    # 套用基線校準偏置修正 (完全對齊 draw_EMG)
                    corrected_infer = batch_data - self.channel_offsets
                    self.inference_buffer.append(corrected_infer)
                    current_len = sum(b.shape[1] for b in self.inference_buffer)
                    if current_len >= self.inference_batch_size:
                        combined_data = np.concatenate(self.inference_buffer, axis=1)
                        if self.data_queue.qsize() > 1:
                            try:
                                self.data_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.data_queue.put(combined_data)
                        self.inference_buffer = []

                    # ── 同步發送 ground-truth 音訊 chunk ──
                    if self.gt_audio is not None and self.gt_audio_idx < self.gt_audio_length:
                        # EMG 689.06 Hz → 對應時間 → 對應 22050 Hz 音訊樣本數
                        emg_duration = batch_data.shape[1] / 689.06
                        audio_chunk_size = int(emg_duration * self.playback_sr)
                        audio_end = min(self.gt_audio_idx + audio_chunk_size, self.gt_audio_length)
                        audio_chunk = self.gt_audio[self.gt_audio_idx:audio_end]
                        self.gt_audio_idx = audio_end
                        if len(audio_chunk) > 0:
                            self.ground_truth_audio_signal.emit(audio_chunk)

                    self.was_recording = True
                else:
                    batch_data = np.random.randn(8, self.ui_batch_size) * 0.15
                    if getattr(self, 'was_recording', False):
                        while not self.data_queue.empty():
                            try:
                                self.data_queue.get_nowait()
                            except queue.Empty:
                                break
                        self.data_queue.put(np.zeros((8, self.inference_batch_size)))
                        self.inference_buffer = []
                        crossfade_remainder = None
                        cf_offset = 0
                        self.was_recording = False

                self.update_plot_signal.emit(batch_data)
                self.msleep(self.fps_delay)

        @Slot(bool)
        def set_recording(self, state):
            self.is_recording = state

        def stop(self):
            self.is_running = False

    class InferenceWorker(QThread):
        result_signal = Signal(str, float)
        mel_signal = Signal(np.ndarray)
        audio_signal = Signal(np.ndarray)

        def __init__(self, data_queue, weight_path=None):
            super().__init__()
            self.data_queue = data_queue
            self.weight_path = weight_path or FLAGS.model_path
            self.is_running = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.cpp_engine = None
            self.streaming_vocoder = None
            self.streaming_state = None
            self.mfcc_norm = None

        def run(self):
            self.result_signal.emit("Status: Loading AI Models...", 0.0)
            
            if os.path.exists('normalizers.pkl'):
                try:
                    import pickle
                    self.mfcc_norm, _ = pickle.load(open('normalizers.pkl', 'rb'))
                except Exception as e:
                    logging.warning(f"Failed to load normalizers.pkl: {e}")

            try:
                if FLAGS.use_cpp:
                    try:
                        from cpp_forward import CppInferenceEngine
                        self.cpp_engine = CppInferenceEngine()
                        self.cpp_engine.load_weights(FLAGS.weights_dir)
                        self.cpp_engine.init_state()
                        logging.info(f"Loaded C++ Engine from {FLAGS.weights_dir}")
                    except Exception as e:
                        logging.warning(f"C++ Engine unavailable ({e}), falling back to PyTorch Model...")
                        self.cpp_engine = None

                if self.cpp_engine is None:
                    n_phones = len(phoneme_inventory)
                    self.model = Model(8, 80, n_phones).to(self.device)

                    candidate_paths = [
                        self.weight_path,
                        'transduction_model.pt',
                        'models_mamba1/transduction_model/model.pt',
                        'models/transduction_model/model.pt',
                        'model.pt'
                    ]
                    actual_path = None
                    for p in candidate_paths:
                        if p and os.path.exists(p):
                            actual_path = p
                            break

                    if FLAGS.use_int8 and os.path.exists(FLAGS.int8_model_path):
                        self.model = _load_int8_quantized_model(FLAGS.int8_model_path, self.model, self.device)
                    elif actual_path:
                        checkpoint = torch.load(actual_path, map_location=self.device)
                        checkpoint = _convert_qat_state_dict(checkpoint)
                        self.model.load_state_dict(checkpoint, strict=False)
                        self.model.eval()
                        logging.info(f"Loaded PyTorch Model from {actual_path}")
                    else:
                        logging.warning(f"Model checkpoint not found in candidate paths: {candidate_paths}")

                    if hasattr(self.model, 'init_streaming_state'):
                        try:
                            self.streaming_state = self.model.init_streaming_state(self.device)
                        except Exception as e:
                            logging.warning(f"Could not init streaming state: {e}")
                            self.streaming_state = None
            except Exception as e:
                err_text = f"Model init error: {e}"
                logging.error(err_text)
                self.result_signal.emit(err_text, 0.0)

            try:
                base_vocoder = Vocoder(device=str(self.device))
                self.streaming_vocoder = StreamingVocoder(base_vocoder)
                logging.info("Streaming Vocoder loaded successfully!")
            except Exception as e:
                logging.error(f"Vocoder error: {e}")

            self.result_signal.emit("Status: Waiting for input...", 0.0)

            while self.is_running:
                try:
                    emg_data = self.data_queue.get(timeout=0.1)
                    
                    if np.all(emg_data == 0):
                        mel_features = np.zeros((emg_data.shape[1], 80))
                        self.mel_signal.emit(mel_features)
                        
                        if self.streaming_vocoder is not None:
                            tail_audio = self.streaming_vocoder.flush()
                            if len(tail_audio) > 0:
                                self.audio_signal.emit(tail_audio)
                            self.audio_signal.emit(np.zeros(2000))

                        if self.cpp_engine is not None:
                            try:
                                self.cpp_engine.init_state()
                            except Exception:
                                pass
                        if self.model is not None and hasattr(self.model, 'init_streaming_state'):
                            try:
                                self.streaming_state = self.model.init_streaming_state(self.device)
                            except Exception:
                                pass
                        continue

                    input_tensor = torch.from_numpy(emg_data.T).float().unsqueeze(0).to(self.device)
                    
                    if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    
                    pred = None
                    with torch.no_grad():
                        if self.cpp_engine is not None:
                            chunk_len = emg_data.shape[1]
                            flat_emg = emg_data.T.flatten().tolist()
                            mel_list = self.cpp_engine.forward_streaming(flat_emg, chunk_len)
                            mel_tensor = torch.tensor(mel_list, dtype=torch.float32).unsqueeze(0)
                            pred = mel_tensor
                        elif self.model is not None:
                            if hasattr(self.model, 'forward_streaming') and self.streaming_state is not None:
                                mel, aux, self.streaming_state = self.model.forward_streaming(input_tensor, self.streaming_state)
                                pred = (mel, aux) if aux is not None else mel
                            else:
                                session_ids = torch.zeros((1,), dtype=torch.long).to(self.device)
                                pred = self.model(None, input_tensor, session_ids)
                    
                    if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - start_time
                    
                    if pred is not None:
                        if isinstance(pred, tuple):
                            mel_tensor = pred[0].detach().cpu()
                            phoneme_pred = pred[1] if len(pred) > 1 else None
                            mel_features = mel_tensor.squeeze(0).numpy() if mel_tensor.dim() == 3 else mel_tensor.numpy()
                        else:
                            mel_tensor = pred.detach().cpu()
                            phoneme_pred = None
                            mel_features = mel_tensor.squeeze(0).numpy() if mel_tensor.dim() == 3 else mel_tensor.numpy()
                        
                        if phoneme_pred is not None:
                            phone_ids = phoneme_pred.argmax(-1).squeeze(0).tolist() if phoneme_pred.dim() >= 2 else phoneme_pred.tolist()
                            if isinstance(phone_ids, int): phone_ids = [phone_ids]
                            text_result = f"Predicted Phonemes Length: {len(phone_ids)}"
                        else:
                            text_result = f"Feature Shape: {pred[0].shape if isinstance(pred, tuple) else pred.shape}"
                        
                        self.result_signal.emit(text_result, latency)
                        self.mel_signal.emit(mel_features)

                        if self.streaming_vocoder is not None:
                            mel_chunk = mel_tensor.squeeze(0)
                            if self.mfcc_norm is not None:
                                if isinstance(mel_chunk, torch.Tensor):
                                    mel_np = mel_chunk.detach().cpu().numpy()
                                    mel_np = self.mfcc_norm.inverse(mel_np)
                                    mel_chunk = torch.from_numpy(mel_np).float()
                                else:
                                    mel_chunk = self.mfcc_norm.inverse(mel_chunk)
                            mel_chunk = mel_chunk.to(self.device)
                            audio_chunk = self.streaming_vocoder.synthesize_chunk(mel_chunk)
                            if len(audio_chunk) > 0:
                                self.audio_signal.emit(audio_chunk)

                except queue.Empty:
                    self.msleep(10)
                    continue
                except Exception as e:
                    err_msg = f"Inference Error: {e}"
                    logging.error(err_msg)
                    self.result_signal.emit(err_msg, 0.0)
                    self.msleep(50)

        def stop(self):
            self.is_running = False

    class MainWindow(QMainWindow):
        recording_signal = Signal(bool)

        def __init__(self):
            super().__init__()
            self.setWindowTitle("8-Channel EMG & Audio System (Realtime Inference)")
            self.resize(1400, 750)
            self.setStyleSheet("background-color: #2b2b2b;")

            main_layout = QVBoxLayout()
            plot_layout = QHBoxLayout()

            # --- 左側：EMG 波形圖 ---
            self.plot_widget = pg.PlotWidget(title="Real-time 8-Channel EMG")
            self.plot_widget.enableAutoRange(axis='y') 
            self.plot_widget.setDownsampling(mode='peak', auto=True)
            self.plot_widget.setClipToView(True)
            plot_layout.addWidget(self.plot_widget, stretch=1)

            self.legend = pg.LegendItem()
            self.plot_widget.getPlotItem().layout.addItem(self.legend, 1, 2)

            colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
                      (255, 100, 255), (100, 255, 255), (255, 150, 50),  (200, 200, 200)]

            self.curves = []
            for i in range(8):
                curve = self.plot_widget.plot(pen=pg.mkPen(color=colors[i], width=1.5))
                self.curves.append(curve)

            for i in range(7, -1, -1):
                self.legend.addItem(self.curves[i], f"Ch {i+1}")

            # --- 右側：模型輸出 (上下分割) ---
            right_layout = QVBoxLayout()

            # 上半部：梅爾頻譜圖
            self.mel_widget = pg.PlotWidget(title="Mel-Spectrogram")
            self.mel_image = pg.ImageItem()
            self.mel_widget.addItem(self.mel_image)
            self.mel_image.setColorMap(pg.colormap.get('viridis'))
            self.mel_image.setLevels([0, 5]) 
            right_layout.addWidget(self.mel_widget, stretch=1)

            # 下半部：音訊波形圖
            self.audio_widget = pg.PlotWidget(title="Vocoder Output")
            self.audio_widget.setYRange(-1.0, 1.0) 
            self.audio_widget.setDownsampling(mode='peak', auto=True)
            self.audio_widget.setClipToView(True)
            self.audio_curve = self.audio_widget.plot(pen=pg.mkPen(color=(0, 255, 255), width=1.5))
            right_layout.addWidget(self.audio_widget, stretch=1)

            plot_layout.addLayout(right_layout, stretch=1)
            main_layout.addLayout(plot_layout)

            # --- 通道品質指示面板 (手機信號風格) ---
            self.quality_analyzer = ChannelQualityAnalyzer(
                num_channels=8, sample_rate=689.06, calibration_duration=1.0
            ) if HAS_CHANNEL_QUALITY else None

            quality_frame = QFrame()
            quality_frame.setStyleSheet(
                "QFrame { background-color: #333333; border: 1px solid #555555;"
                " border-radius: 6px; padding: 4px; }"
            )
            quality_layout = QHBoxLayout(quality_frame)
            quality_layout.setContentsMargins(10, 6, 10, 6)
            quality_title = QLabel("Channel Quality:")
            quality_title.setStyleSheet("color: #aaaaaa; font-size: 13px; font-weight: bold; border: none;")
            quality_layout.addWidget(quality_title)
            quality_layout.addStretch(1)

            self.signal_bars = []
            for i in range(8):
                bar_widget = SignalBarsWidget(i)
                bar_widget.setToolTip(f"{CH_LABELS[i] if i < len(CH_LABELS) else f'Ch{i+1}'}\nUncalibrated (Click Calibrate to establish baseline)")
                self.signal_bars.append(bar_widget)
                quality_layout.addWidget(bar_widget)
            quality_layout.addStretch(1)

            # Quick calibrate button next to quality panel
            self.btn_quick_calib = QPushButton("Calibrate EMG Signal")
            self.btn_quick_calib.setToolTip("Collect 1-second resting EMG signal to calibrate baseline [C]")
            self.btn_quick_calib.setStyleSheet(
                "QPushButton { background-color: #444444; color: #5bc0de; border: 1px solid #5bc0de; border-radius: 4px; padding: 5px 12px; font-size: 12px; font-weight: bold; }"
                "QPushButton:hover { background-color: #5bc0de; color: #ffffff; }"
                "QPushButton:disabled { color: #888888; border-color: #666666; background-color: #333333; }"
            )
            self.btn_quick_calib.clicked.connect(self.start_calibration)
            quality_layout.addWidget(self.btn_quick_calib)

            main_layout.addWidget(quality_frame)

            # --- 校準狀態與各通道偏置變數 (完全符合 draw_EMG 規範) ---
            self.is_calibrating = False
            self.calib_duration_sec = 1.0  # 1 秒鐘靜息採集
            self.calib_samples_needed = int(689.06 * self.calib_duration_sec)
            self.calib_samples_collected = 0
            self.calib_buffer = [[] for _ in range(8)]
            self.channel_offsets = np.zeros(8, dtype=np.float32)
            self.channel_stds = np.zeros(8, dtype=np.float32)

            # --- 按鈕列 (推論擷取 + 專用基線校準按鈕) ---
            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(12)

            self.btn_ptt = QPushButton("Click to Start Capture")
            self.btn_ptt.setStyleSheet(
                "QPushButton { background-color: #5cb85c; color: white; font-size: 18px; font-weight: bold; padding: 14px; border-radius: 6px; }"
                "QPushButton:hover { background-color: #4cae4c; }"
                "QPushButton:disabled { background-color: #666666; color: #aaaaaa; }"
            )
            self.btn_ptt.clicked.connect(self.toggle_recording)
            btn_layout.addWidget(self.btn_ptt, stretch=3)

            self.btn_calib = QPushButton("Calibrate EMG Signal")
            self.btn_calib.setToolTip("Collecting Silent EMG Signal for Calibration")
            self.btn_calib.setStyleSheet(
                "QPushButton { background-color: #0275d8; color: white; font-size: 18px; font-weight: bold; padding: 14px; border-radius: 6px; }"
                "QPushButton:hover { background-color: #025aa5; }"
                "QPushButton:disabled { background-color: #f0ad4e; color: #000000; }"
            )
            self.btn_calib.clicked.connect(self.start_calibration)
            btn_layout.addWidget(self.btn_calib, stretch=2)

            main_layout.addLayout(btn_layout)
            
            self.app_state = 'ready'  # ready | recording
            self.is_recording = False
            self.label_result = QLabel("System Initializing...")
            self.label_result.setStyleSheet("font-size: 20px; color: #ffffff;")
            main_layout.addWidget(self.label_result)

            container = QWidget()
            container.setLayout(main_layout)
            self.setCentralWidget(container)

            # 緩衝區設定
            self.buffer_size = 2000
            self.plot_data = np.zeros((8, self.buffer_size))
            
            self.mel_buffer_size = 500
            self.mel_data = np.zeros((self.mel_buffer_size, 80))

            self.audio_buffer_size = 33000 
            self.audio_data = np.zeros(self.audio_buffer_size)
            
            self.inference_queue = queue.Queue()

            # ── 音訊播放設定 (sounddevice) ──
            self.audio_sr = 22050
            self._audio_playback_buf = np.zeros(0, dtype=np.float32)
            self._audio_lock = threading.Lock()
            self.audio_stream = None
            if HAS_SOUNDDEVICE:
                try:
                    self.audio_stream = sd.OutputStream(
                        samplerate=self.audio_sr,
                        channels=1,
                        dtype='float32',
                        blocksize=1024,
                        latency='low',
                        callback=self._audio_callback,
                    )
                    self.audio_stream.start()
                    logging.info("[Audio] OutputStream started")
                except Exception as e:
                    logging.warning(f"[Audio] Failed to open audio device: {e}")
                    self.audio_stream = None

            import shutil, subprocess
            self.pacat_proc = None
            if (self.audio_stream is None or not self.audio_stream.active) and shutil.which('pacat') is not None:
                try:
                    self.pacat_proc = subprocess.Popen(
                        ['pacat', '--format=s16le', '--channels=1',
                         f'--rate={self.audio_sr}', '--raw',
                         '--stream-name=EMG-to-Speech', '--latency-msec=100'],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL
                    )
                    logging.info("[Audio] pacat live streaming process started (PulseAudio fallback)")
                except Exception as e:
                    logging.warning(f"[Audio] Failed to start pacat: {e}")

            weight_file_path = FLAGS.model_path
            self.inference_thread = InferenceWorker(self.inference_queue, weight_file_path)
            self.inference_thread.result_signal.connect(self.update_result_ui)
            self.inference_thread.mel_signal.connect(self.update_mel_ui)
            self.inference_thread.audio_signal.connect(self.update_audio_ui)
            self.inference_thread.start()

            emg_search_root = FLAGS.emg_file or "emg_data"
            self.playback_thread = FilePlaybackWorker(
                emg_search_root, self.inference_queue, crossfade_ms=200
            )
            self.playback_thread.update_plot_signal.connect(self.update_plot_ui)
            self.playback_thread.file_changed_signal.connect(self.update_file_info)
            self.playback_thread.status_signal.connect(self.update_status_label)
            self.playback_thread.ground_truth_audio_signal.connect(self.play_ground_truth_audio)
            self.recording_signal.connect(self.playback_thread.set_recording)
            self.playback_thread.start()

        @Slot()
        def start_calibration(self):
            """觸發 8 通道基線校準 (完全符合 draw_EMG 的 start_calibration 規範)"""
            if self.is_calibrating:
                return

            # 若處於錄製狀態，先暫停錄製以獲取純淨靜息基線
            if self.is_recording:
                self.toggle_recording()

            self.is_calibrating = True
            self.calib_samples_collected = 0
            self.calib_buffer = [[] for _ in range(8)]

            if self.quality_analyzer:
                self.quality_analyzer.start_calibration()

            # Update buttons and status label
            self.btn_ptt.setEnabled(False)
            self.btn_calib.setEnabled(False)
            self.btn_calib.setText("Calibrating EMG Signal (0%)... Please relax.")
            self.btn_calib.setStyleSheet(
                "QPushButton { background-color: #f0ad4e; color: #000000; font-size: 18px; font-weight: bold; padding: 14px; border-radius: 6px; }"
            )
            if hasattr(self, 'btn_quick_calib'):
                self.btn_quick_calib.setEnabled(False)
                self.btn_quick_calib.setText("Calibrating...")

            self.label_result.setText("Start Calibrating... Please relax your face for 1 second.")
            self.label_result.setStyleSheet("font-size: 20px; color: #f0ad4e; font-weight: bold;")
            logging.info("[Calibration] Please relax facial muscles completely for 1 second...")
            print("\n>> [Calibration] Please relax facial muscles completely for 1 second...")

        def finish_calibration(self):
            """Complete 8-channel baseline calibration and compute per-channel offsets."""
            self.is_calibrating = False

            for ch in range(8):
                if self.calib_buffer[ch]:
                    pts = np.array(self.calib_buffer[ch][:self.calib_samples_needed], dtype=np.float32)
                    self.channel_offsets[ch] = float(np.mean(pts))
                    self.channel_stds[ch] = float(np.std(pts))
                else:
                    self.channel_offsets[ch] = 0.0
                    self.channel_stds[ch] = 0.0

            # Finish quality analyzer calibration
            if self.quality_analyzer:
                self.quality_analyzer.finish_calibration()

            # Synchronize offsets with FilePlaybackWorker for inference
            if hasattr(self, 'playback_thread') and hasattr(self.playback_thread, 'set_channel_offsets'):
                self.playback_thread.set_channel_offsets(self.channel_offsets)

            # Update legend text with calibrated offsets
            self.legend.clear()
            for i in range(7, -1, -1):
                self.legend.addItem(self.curves[i], f"Ch {i+1} (Offset={self.channel_offsets[i]:+.2f})")

            # Update SignalBarsWidget tooltips with English descriptions
            for i in range(8):
                label = CH_LABELS[i] if i < len(CH_LABELS) else f"Ch {i+1}"
                self.signal_bars[i].setToolTip(
                    f"{label}\nDC Offset: {self.channel_offsets[i]:+.3f}\nResting Noise (STD): {self.channel_stds[i]:.3f}"
                )

            # Restore button states
            self.btn_ptt.setEnabled(True)
            self.btn_calib.setEnabled(True)
            self.btn_calib.setText("Re-calibrate EMG Signal")
            self.btn_calib.setStyleSheet(
                "QPushButton { background-color: #0275d8; color: white; font-size: 18px; font-weight: bold; padding: 14px; border-radius: 6px; }"
                "QPushButton:hover { background-color: #025aa5; }"
            )
            if hasattr(self, 'btn_quick_calib'):
                self.btn_quick_calib.setEnabled(True)
                self.btn_quick_calib.setText("Re-calibrate EMG Signal")

            offset_str = ", ".join([f"Ch{i+1}:{self.channel_offsets[i]:+.2f}" for i in range(8)])
            self.label_result.setText(f"Reset Baseline ({offset_str})")
            self.label_result.setStyleSheet("font-size: 17px; color: #5cb85c; font-weight: bold;")
            logging.info(f"[+] Calibration complete! Channel offsets updated: {offset_str}")
            print(f"\n[+] Calibration complete! Channel offsets updated:\n    {offset_str}")
            print("    Baseline zeroed. Ready to capture EMG signals.\n")

        def keyPressEvent(self, event):
            key = event.key()
            if key == Qt.Key_C:
                self.start_calibration()
            elif key == Qt.Key_R or key == Qt.Key_Space:
                self.toggle_recording()
            else:
                super().keyPressEvent(event)

        def toggle_recording(self, checked=False):
            if self.is_calibrating:
                return
            self.is_recording = not self.is_recording
            if self.is_recording:
                self.app_state = 'recording'
                self.btn_ptt.setText(" Click to Stop Capture")
                self.btn_ptt.setStyleSheet(
                    "QPushButton { background-color: #d9534f; color: white; font-size: 18px; font-weight: bold; padding: 14px; border-radius: 6px; }"
                    "QPushButton:hover { background-color: #c9302c; }"
                )
                self.label_result.setText("Status: Capturing EMG signals...")
                self.label_result.setStyleSheet("font-size: 20px; color: #ffffff;")
                self.recording_signal.emit(True)
            else:
                self.app_state = 'ready'
                self.btn_ptt.setText(" Click to Start Capture")
                self.btn_ptt.setStyleSheet(
                    "QPushButton { background-color: #5cb85c; color: white; font-size: 18px; font-weight: bold; padding: 14px; border-radius: 6px; }"
                    "QPushButton:hover { background-color: #4cae4c; }"
                )
                self.label_result.setText("Status: Stopped capturing.")
                self.label_result.setStyleSheet("font-size: 20px; color: #ffffff;")
                self.recording_signal.emit(False)

        @Slot(str)
        def update_status_label(self, msg):
            self.label_result.setText(msg)

        @Slot(np.ndarray)
        def update_plot_ui(self, new_data):
            # 1. Calibration phase: accumulate resting baseline samples
            if self.is_calibrating:
                n = new_data.shape[1]
                for ch in range(8):
                    self.calib_buffer[ch].extend(new_data[ch].tolist())
                self.calib_samples_collected += n

                if self.quality_analyzer:
                    self.quality_analyzer.feed_calibration_data(new_data)

                pct = min(100, int(self.calib_samples_collected / self.calib_samples_needed * 100))
                self.btn_calib.setText(f"Calibrating EMG Signal ({pct}%)... Please relax.")
                if hasattr(self, 'btn_quick_calib'):
                    self.btn_quick_calib.setText(f"Calibrating ({pct}%)...")
                self.label_result.setText(f"Calibrating EMG Signal ({pct}%)... Please relax.")

                if self.calib_samples_collected >= self.calib_samples_needed:
                    self.finish_calibration()

            # 2. Subtract calibrated baseline offset (x = raw - offset)
            corrected_data = new_data - self.channel_offsets[:, np.newaxis]

            shift = corrected_data.shape[1]
            self.plot_data[:, :-shift] = self.plot_data[:, shift:]
            self.plot_data[:, -shift:] = corrected_data
            
            for i in range(8):
                offset = (i - 3.5) * 2
                self.curves[i].setData(self.plot_data[i] + offset)

            if self.quality_analyzer:
                statuses = self.quality_analyzer.analyze_chunk(corrected_data)
                for st in statuses:
                    self.signal_bars[st.channel_id].set_status(st)

        @Slot(np.ndarray)
        def update_mel_ui(self, new_mel):
            shift = new_mel.shape[0]
            if shift == 0: return
            self.mel_data[:-shift, :] = self.mel_data[shift:, :]
            self.mel_data[-shift:, :] = new_mel
            self.mel_image.setImage(self.mel_data, autoLevels=False)

        @Slot(np.ndarray)
        def update_audio_ui(self, new_audio):
            """更新音訊波形圖 (僅視覺化，實際播放改用 ground-truth)"""
            shift = len(new_audio)
            if shift == 0: 
                return
            
            new_audio = new_audio.astype(np.float32).copy()

            # ── 區塊邊界平滑過渡 (64 採樣點 / ~3ms Crossfade) ──
            # 消除 Vocoder 各 chunk 邊界相位的突變跳變與喀噠雜音 (Clicks & Pops)
            if hasattr(self, '_prev_audio_tail') and self._prev_audio_tail is not None and len(self._prev_audio_tail) > 0:
                fade_len = min(64, shift, len(self._prev_audio_tail))
                if fade_len > 0:
                    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    new_audio[:fade_len] = (
                        self._prev_audio_tail[-fade_len:] * fade_out + new_audio[:fade_len] * fade_in
                    )
            self._prev_audio_tail = new_audio.copy()

            # 幅度限制與平滑 tanh 軟剪裁，防止數位破音爆音
            new_audio = np.tanh(new_audio * 1.1)

            if shift >= self.audio_buffer_size:
                self.audio_data = new_audio[-self.audio_buffer_size:]
            else:
                self.audio_data[:-shift] = self.audio_data[shift:]
                self.audio_data[-shift:] = new_audio
                
            visual_data = self.audio_data[::80]
            self.audio_curve.setData(visual_data)
            # 注意: vocoder 音訊不再播放，改由 play_ground_truth_audio 處理

        @Slot(np.ndarray)
        def play_ground_truth_audio(self, audio_chunk):
            """播放 ground-truth 音訊 (已 resample 至 22050 Hz)"""
            if len(audio_chunk) == 0:
                return
            audio_chunk = audio_chunk.astype(np.float32).copy()

            if HAS_SOUNDDEVICE and self.audio_stream is not None and self.audio_stream.active:
                with self._audio_lock:
                    self._audio_playback_buf = np.concatenate([self._audio_playback_buf, audio_chunk])
                    max_buf = self.audio_sr * 2
                    if len(self._audio_playback_buf) > max_buf:
                        self._audio_playback_buf = self._audio_playback_buf[-max_buf:]
            elif getattr(self, 'pacat_proc', None) is not None:
                try:
                    pcm_bytes = (audio_chunk * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
                    self.pacat_proc.stdin.write(pcm_bytes)
                    self.pacat_proc.stdin.flush()
                except Exception:
                    pass

        def _audio_callback(self, outdata, frames, time_info, status):
            with self._audio_lock:
                available = len(self._audio_playback_buf)
                if available >= frames:
                    outdata[:, 0] = self._audio_playback_buf[:frames]
                    self._audio_playback_buf = self._audio_playback_buf[frames:]
                elif available > 0:
                    outdata[:available, 0] = self._audio_playback_buf
                    outdata[available:, 0] = 0.0
                    self._audio_playback_buf = np.zeros(0, dtype=np.float32)
                else:
                    outdata[:, 0] = 0.0

        @Slot(str, int, int)
        def update_file_info(self, filename, current, total):
            self.setWindowTitle(f"8-Channel EMG & Audio System — [{current}/{total}] {filename}")

        @Slot(str, float)
        def update_result_ui(self, text, latency):
            if "Loading" in text or "Waiting" in text:
                self.label_result.setText(text)
            else:
                self.label_result.setText(f"Result: {text} (Latency: {latency:.2f}s)")



        def closeEvent(self, event):
            self.playback_thread.stop()
            self.inference_thread.stop()
            self.playback_thread.wait()
            self.inference_thread.wait()
            if HAS_SOUNDDEVICE and self.audio_stream and self.audio_stream.active:
                self.audio_stream.stop()
                self.audio_stream.close()
            event.accept()


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


def _convert_qat_state_dict(state_dict):
    """
    將 QAT checkpoint 的 state_dict key 轉換回原始 architecture.Model 格式。

    QAT 訓練會做兩件事:
      1. Conv1d 包成 QuantizedConv1d → key 多一層 .conv.
         Linear 包成 QuantizedLinear → key 多一層 .linear.
      2. ResBlock 之間插入 ActivationQuantHook → conv_blocks 索引變成 0,2,4 (原本 0,1,2)

    此函數把 QAT key 轉回原始格式, 同時也能透通處理非 QAT 的 checkpoint。
    """
    # 快速偵測: 如果沒有任何 QAT 特徵 key, 直接回傳
    has_qat_keys = any(
        '.conv.weight' in k or '.conv.bias' in k or
        '.linear.weight' in k or '.linear.bias' in k
        for k in state_dict.keys()
    )
    if not has_qat_keys:
        return state_dict

    logging.info('  偵測到 QAT checkpoint, 自動轉換 state_dict keys...')

    # conv_blocks 索引重映射: QAT 插入 ActivationQuantHook 後
    # 原本 [ResBlock0, ResBlock1, ResBlock2] → [ResBlock0, ActQuant, ResBlock1, ActQuant, ResBlock2, ActQuant]
    # 即 conv_blocks.{0,2,4} → conv_blocks.{0,1,2}
    qat_to_orig_idx = {0: 0, 2: 1, 4: 2}

    new_state_dict = {}
    skipped = 0
    for key, value in state_dict.items():
        new_key = key

        # 跳過 ActivationQuantHook 的參數 (它們在原始模型中不存在)
        # ActivationQuantHook 是 conv_blocks.{1,3,5}
        if key.startswith('conv_blocks.'):
            parts = key.split('.')
            idx = int(parts[1])
            if idx in (1, 3, 5):
                skipped += 1
                continue
            # 重映射索引
            if idx in qat_to_orig_idx:
                parts[1] = str(qat_to_orig_idx[idx])
                new_key = '.'.join(parts)

        # 去掉 QuantizedConv1d 的 .conv. 包裝
        new_key = new_key.replace('.conv1.conv.', '.conv1.')
        new_key = new_key.replace('.conv2.conv.', '.conv2.')
        new_key = new_key.replace('.residual_path.conv.', '.residual_path.')

        # 去掉 QuantizedLinear 的 .linear. 包裝
        new_key = new_key.replace('.linear.weight', '.weight')
        new_key = new_key.replace('.linear.bias', '.bias')

        # 跳過 num_batches_tracked (原始模型不追蹤)
        if new_key.endswith('.num_batches_tracked'):
            skipped += 1
            continue

        new_state_dict[new_key] = value

    logging.info(f'  轉換完成: {len(new_state_dict)} 個 key (跳過 {skipped} 個 QAT 專用 key)')
    return new_state_dict


def _load_int8_quantized_model(int8_path, model, device):
    """
    載入 INT8 量化模型 (qat_model_int8.pt) 並反量化為 FP32 載入到 Model。

    INT8 檔案格式 (由 qat_int8.py export_int16_weights 匯出):
      {
        'quantized_weights': {name: int8_tensor},
        'scales':            {name: (scale, zero_point)},
        'other_params':      {name: fp32_tensor},
        'bits':              8,
        'per_channel':       True,
      }

    反量化公式: weight_fp32 = (int8_weight - zero_point) * scale
    """
    logging.info(f'  載入 INT8 量化模型: {int8_path}')
    ckpt = torch.load(int8_path, map_location=device, weights_only=False)

    quantized_weights = ckpt['quantized_weights']
    scales = ckpt['scales']
    other_params = ckpt['other_params']
    bits = ckpt.get('bits', 8)
    per_channel = ckpt.get('per_channel', True)

    logging.info(f'  INT8 模型: {len(quantized_weights)} 個量化權重, '
                 f'{len(other_params)} 個非量化參數, bits={bits}')

    # ── 反量化 INT8 → FP32 ────────────────────────────────────
    dequantized_state = {}
    for name, q_tensor in quantized_weights.items():
        scale, zero_point = scales[name]
        # reshape scale/zero_point for broadcast
        shape = [q_tensor.size(0)] + [1] * (q_tensor.dim() - 1)
        s = scale.view(shape).to(device)
        zp = zero_point.view(shape).to(device)
        # 反量化: fp32 = (int8 - zero_point) * scale
        w_fp32 = (q_tensor.to(device).float() - zp) * s
        dequantized_state[name] = w_fp32

    # ── 合併為完整 state_dict ─────────────────────────────────
    full_state_dict = {}
    full_state_dict.update(dequantized_state)
    for name, param in other_params.items():
        full_state_dict[name] = param.to(device)

    # ── QAT key 轉換 (與 FP32 QAT 模型相同的 key 修正) ────────
    full_state_dict = _convert_qat_state_dict(full_state_dict)

    # ── 載入到模型 ────────────────────────────────────────────
    model.load_state_dict(full_state_dict)
    model.eval()

    # ── 統計資訊 ──────────────────────────────────────────────
    q_size = sum(t.nelement() * 1 for t in quantized_weights.values())
    o_size = sum(t.nelement() * 4 for t in other_params.values())
    s_size = sum(s.nelement() * 4 + z.nelement() * 4 for s, z in scales.values())
    int8_mb = (q_size + o_size + s_size) / 1024 / 1024
    fp32_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
    logging.info(f'  INT8 檔案大小: {int8_mb:.1f} MB → 反量化後 FP32: {fp32_mb:.1f} MB')
    logging.info(f'  INT8 模型載入完成')

    return model


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

    # ── 啟動 GUI 介面 ──────────────────────────────────────────
    if FLAGS.gui:
        if not HAS_PYSIDE6:
            logging.error("PySide6 or pyqtgraph is not installed. Cannot start GUI! Please install: pip install PySide6 pyqtgraph")
            sys.exit(1)

        logging.info("Starting PySide6 GUI interface...")
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

    # ── 載入資料集 (CLI 模式) ──────────────────────────────────
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
        logging.info(f'每段音訊將存到: {chunk_dir}/')

    if FLAGS.use_cpp:
        # ── C++ 推論引擎 ──────────────────────────────────────
        logging.info(f'載入 C++ 推論引擎: {FLAGS.weights_dir}')
        from cpp_forward import CppInferenceEngine
        cpp_engine = CppInferenceEngine()
        cpp_engine.load_weights(FLAGS.weights_dir)

        pipeline = CppStreamingPipeline(
            cpp_engine=cpp_engine,
            streaming_vocoder=streaming_vocoder,
            audio_normalizer=devset.mfcc_norm,
            device=device,
            vocoder_chunk_size=FLAGS.vocoder_chunk_size,
            play_mode=FLAGS.play_mode,
            chunk_output_dir=chunk_dir,
        )
        engine_label = 'C++ Engine'
    else:
        # ── PyTorch 模型 ──────────────────────────────────────
        n_phones = len(phoneme_inventory)
        model = Model(devset.num_features, devset.num_speech_features,
                      n_phones).to(device)

        if FLAGS.use_int8:
            # ── INT8 量化模型 ─────────────────────────────────
            logging.info(f'載入 INT8 量化模型: {FLAGS.int8_model_path}')
            model = _load_int8_quantized_model(
                FLAGS.int8_model_path, model, device
            )
            engine_label = 'PyTorch Model (INT8 QAT)'
        else:
            # ── 標準 FP32 / QAT FP32 模型 ────────────────────
            logging.info(f'載入模型: {FLAGS.model_path}')
            state_dict = torch.load(FLAGS.model_path, map_location=device)
            state_dict = _convert_qat_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model.eval()
            engine_label = 'PyTorch Model'

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
    logging.info('=' * 60)
    logging.info(f' 啟動多執行緒即時 EMG → 語音 Pipeline ({engine_label})')
    logging.info('  Thread 1: EMG 擷取')
    logging.info('  Thread 2: 前處理')
    if FLAGS.use_cpp:
        logging.info('  Thread 3: 推論 (C++ CNN+Mamba → Vocoder)')
    else:
        logging.info('  Thread 3: 推論 (因果 CNN → Stateful Mamba → Vocoder)')
    logging.info(f'  播放模式: {FLAGS.play_mode}')
    logging.info('=' * 60)

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

        logging.info('=' * 60)
        logging.info(f'   推論時間:   {infer_time:.2f}s')
        logging.info(f'   總時間:     {total_time:.2f}s (含播放)')
        logging.info(f'   音訊長度:   {audio_dur:.2f}s')
        logging.info(f'   推論 RTF:   {infer_time / audio_dur:.3f}x')

        if FLAGS.save_wav:
            sf.write(FLAGS.streaming_output, full_audio, AUDIO_SR)
            logging.info(f'WAV 已儲存: {FLAGS.streaming_output}')
        logging.info('=' * 60)
    else:
        logging.warning(' 未產生任何音訊')
        return

    # ── (可選) 批次推論比較 (僅 PyTorch 模式) ─────────────────
    if FLAGS.compare_batch and sample is not None and not FLAGS.use_cpp:
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
