"""
C++ 加速版即時串流 Pipeline

使用 pybind11 的 cpp_forward 模組做 EMG → Mel 推論 (CNN + Mamba)，
Vocoder 仍用 PyTorch HiFi-GAN。

與 StreamingPipeline 的差異:
    - Thread 2: 不轉 torch tensor, 直接傳 numpy array
    - Thread 3: 用 CppInferenceEngine.forward_streaming() 取代 model.forward_streaming()
    - 不需要 PyTorch model / model_state
"""

import multiprocessing
import threading
import queue
import subprocess
import shutil
import time
import logging
import numpy as np
import os
import soundfile as sf

import torch

AUDIO_SR = 22050
HOP_SIZE = 256
SENTINEL = None
SENTINEL_BYTES = b'__END__'


def _realtime_playback_worker(audio_q, sr):
    """獨立 process: 從 Queue 讀 PCM → pacat 即時播放"""
    if shutil.which('pacat') is None:
        while True:
            msg = audio_q.get()
            if msg == SENTINEL_BYTES:
                break
        return

    try:
        pacat = subprocess.Popen(
            ['pacat', '--format=s16le', '--channels=1',
             f'--rate={sr}', '--raw',
             '--stream-name=EMG-to-Speech',
             '--latency-msec=200'],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        while True:
            msg = audio_q.get()
            if msg == SENTINEL_BYTES:
                break
        return

    try:
        while True:
            msg = audio_q.get()
            if msg == SENTINEL_BYTES:
                break
            pacat.stdin.write(msg)
            pacat.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            pacat.stdin.close()
            pacat.wait(timeout=10)
        except Exception:
            try:
                pacat.kill()
            except Exception:
                pass


class CppStreamingPipeline:
    """使用 C++ 引擎的串流推論 Pipeline"""

    def __init__(self, cpp_engine, streaming_vocoder, audio_normalizer,
                 device, vocoder_chunk_size=10, play_mode='paplay',
                 chunk_output_dir=None):
        """
        Args:
            cpp_engine: CppInferenceEngine instance (已 load_weights)
            streaming_vocoder: StreamingVocoder instance
            audio_normalizer: FeatureNormalizer (有 inverse() 方法)
            device: vocoder 的 device
            vocoder_chunk_size: 每次送 vocoder 的 mel 幀數
            play_mode: 'paplay' | 'realtime' | 'none'
            chunk_output_dir: 每段音訊存檔目錄 (None=不存)
        """
        self.cpp_engine = cpp_engine
        self.streaming_vocoder = streaming_vocoder
        self.audio_normalizer = audio_normalizer
        self.device = device
        self.vocoder_chunk_size = vocoder_chunk_size
        self.play_mode = play_mode
        self.chunk_output_dir = chunk_output_dir

        self.emg_queue = queue.Queue(maxsize=64)
        self.processed_queue = queue.Queue(maxsize=64)

        self.audio_segments = []
        self.total_audio_samples = 0
        self.chunk_count = 0
        self.vocoder_count = 0
        self.inference_done_time = None

        # realtime 播放用
        self._rt_queue = None
        self._rt_proc = None

        if self.chunk_output_dir:
            os.makedirs(self.chunk_output_dir, exist_ok=True)

    def start(self, emg_source):
        """啟動 pipeline, 初始化 C++ streaming state"""
        self.cpp_engine.init_state()

        # 啟動即時播放 process
        if self.play_mode == 'realtime':
            self._rt_queue = multiprocessing.Queue()
            self._rt_proc = multiprocessing.Process(
                target=_realtime_playback_worker,
                args=(self._rt_queue, AUDIO_SR),
                daemon=True,
            )
            self._rt_proc.start()
            logging.info('🔊 即時播放 process 已啟動 (pacat)')

        self._t1 = threading.Thread(target=self._emg_capture_loop,
                                     args=(emg_source,), daemon=True)
        self._t2 = threading.Thread(target=self._preprocess_loop, daemon=True)
        self._t3 = threading.Thread(target=self._inference_loop, daemon=True)

        self._t1.start()
        self._t2.start()
        self._t3.start()

    def wait(self, wav_path=None):
        """等推論完成, 依模式播放"""
        self._t1.join()
        self._t2.join()
        self._t3.join()

        # 結束即時播放
        if self._rt_queue is not None:
            self._rt_queue.put(SENTINEL_BYTES)
        if self._rt_proc is not None:
            self._rt_proc.join(timeout=30)
            if self._rt_proc.is_alive():
                self._rt_proc.kill()
            logging.info('🔊 即時播放結束')

        # paplay 模式: 推論完後播放完整 WAV
        if self.play_mode == 'paplay' and wav_path and len(self.audio_segments) > 0:
            self._play_with_paplay(wav_path)

    def _send_realtime(self, audio):
        """送音訊到即時播放 process (s16le 格式)"""
        if self._rt_queue is not None and len(audio) > 0:
            try:
                audio_i16 = np.clip(audio, -1.0, 1.0)
                audio_i16 = (audio_i16 * 32767).astype(np.int16)
                self._rt_queue.put_nowait(audio_i16.tobytes())
            except Exception:
                pass

    def _play_with_paplay(self, wav_path):
        """用 paplay 播放 WAV — 乾淨無雜音"""
        if shutil.which('paplay') is None:
            logging.warning('找不到 paplay, 無法播放')
            return

        full_audio = self.get_full_audio()
        sf.write(wav_path, full_audio, AUDIO_SR)
        logging.info(f'🔊 開始播放: {wav_path}')

        try:
            proc = subprocess.Popen(
                ['paplay', wav_path],
                stderr=subprocess.DEVNULL,
            )
            proc.wait()
            logging.info('🔊 播放結束')
        except Exception as e:
            logging.warning(f'播放錯誤: {e}')

    def get_full_audio(self):
        if self.audio_segments:
            return np.concatenate(self.audio_segments)
        return np.array([], dtype=np.float32)

    # ── Thread 1: EMG 擷取 (與原版相同) ──────────────────────
    def _emg_capture_loop(self, emg_source):
        try:
            for chunk in emg_source:
                self.emg_queue.put(chunk)
        except Exception as e:
            logging.error(f'EMG 擷取錯誤: {e}')
        finally:
            self.emg_queue.put(SENTINEL)

    # ── Thread 2: 前處理 (不轉 torch, 直接傳 numpy) ─────────
    def _preprocess_loop(self):
        try:
            while True:
                raw = self.emg_queue.get()
                if raw is SENTINEL:
                    break
                # 直接傳 numpy array, 不需要轉 torch tensor
                self.processed_queue.put(raw)
        except Exception as e:
            logging.error(f'前處理錯誤: {e}')
        finally:
            self.processed_queue.put(SENTINEL)

    # ── Thread 3: 推論 (C++ engine + torch vocoder) ──────────
    def _inference_loop(self):
        mel_pending = []

        try:
            while True:
                emg_chunk = self.processed_queue.get()
                if emg_chunk is SENTINEL:
                    break

                self.chunk_count += 1
                t0 = time.time()

                # ── C++ 推論: numpy → list[float] → C++ → list[list[float]] ──
                chunk_len = emg_chunk.shape[0]
                flat_emg = emg_chunk.flatten().tolist()
                mel_list = self.cpp_engine.forward_streaming(flat_emg, chunk_len)

                # ── 轉 torch tensor 做 normalizer inverse ──
                mel_frames = torch.tensor(mel_list, dtype=torch.float32)
                mel_frames = self.audio_normalizer.inverse(mel_frames)
                mel_pending.append(mel_frames)
                total_pending = sum(m.shape[0] for m in mel_pending)

                if total_pending >= self.vocoder_chunk_size:
                    mel_cat = torch.cat(mel_pending, dim=0)
                    mel_pending = []

                    audio = self.streaming_vocoder.synthesize_chunk(
                        mel_cat.to(self.device)
                    )

                    if len(audio) > 0:
                        self.audio_segments.append(audio.copy())
                        self.total_audio_samples += len(audio)
                        self.vocoder_count += 1

                        # 即時播放
                        self._send_realtime(audio)

                        # 存每段音訊
                        if self.chunk_output_dir:
                            chunk_path = os.path.join(
                                self.chunk_output_dir,
                                f'chunk_{self.vocoder_count:03d}.wav'
                            )
                            sf.write(chunk_path, audio, AUDIO_SR)

                        dt = time.time() - t0
                        audio_ms = len(audio) / AUDIO_SR * 1000
                        logging.info(
                            f'Chunk {self.chunk_count:4d} | '
                            f'延遲 {dt*1000:6.1f}ms | '
                            f'🔊 {audio_ms:6.1f}ms | '
                            f'累計 {self.total_audio_samples/AUDIO_SR:.2f}s'
                        )

            # Flush 剩餘 mel
            if mel_pending:
                mel_cat = torch.cat(mel_pending, dim=0)
                audio = self.streaming_vocoder.synthesize_chunk(
                    mel_cat.to(self.device)
                )
                if len(audio) > 0:
                    self.audio_segments.append(audio.copy())
                    self.total_audio_samples += len(audio)
                    self._send_realtime(audio)

            tail = self.streaming_vocoder.flush()
            if len(tail) > 0:
                self.audio_segments.append(tail.copy())
                self.total_audio_samples += len(tail)
                self._send_realtime(tail)

        except Exception as e:
            logging.error(f'推論錯誤: {e}', exc_info=True)

        self.inference_done_time = time.time()
        logging.info(f'✅ C++ 推論完成 ({self.chunk_count} chunks, '
                     f'{self.total_audio_samples/AUDIO_SR:.2f}s 音訊)')
