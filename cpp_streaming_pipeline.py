'''"""
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

                # ── C++ 推論：直接 NumPy -> C++ -> NumPy ─────────────
                # 避免 flatten().tolist() 的 Python object 轉換成本。
                emg_chunk = np.ascontiguousarray(emg_chunk, dtype=np.float32)
                mel_np = self.cpp_engine.forward_streaming(emg_chunk)

                # ── 轉 torch tensor做 normalizer inverse ─────────────
                # from_numpy 不複製資料（必要時後續 tensor op 再處理）。
                mel_frames = torch.from_numpy(
                    np.asarray(mel_np, dtype=np.float32)
                )
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
'''
"""
C++  Pipeline

Uses the pybind11 cpp_forward module for EMG -> Mel inference (CNN + Mamba).
The vocoder remains the PyTorch HiFi-GAN implementation.

Differences from StreamingPipeline:
    - Thread 2: keeps data as NumPy instead of converting to Torch tensors
    - Thread 3: uses CppInferenceEngine.forward_streaming() instead of model.forward_streaming()
    - No PyTorch acoustic model or model_state is required
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
    """Separate process: read PCM from the queue and stream it to pacat."""
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
    """Streaming inference pipeline using the C++ acoustic-model engine."""

    def __init__(self, cpp_engine, streaming_vocoder, audio_normalizer,
                 device, vocoder_chunk_size=10, play_mode='paplay',
                 chunk_output_dir=None, profile_timing=True,
                 profile_every=1):
        """
        Args:
            cpp_engine: CppInferenceEngine instance (already loaded with load_weights())
            streaming_vocoder: StreamingVocoder instance
            audio_normalizer: FeatureNormalizer ( inverse() )
            device: device used by the vocoder
            vocoder_chunk_size: number of mel frames passed to the vocoder per call
            play_mode: 'paplay' | 'realtime' | 'none'
            chunk_output_dir: audio (None=)
            profile_timing: whether to log per-stage timing information
            profile_every: log timing every N C++ chunks
        """
        self.cpp_engine = cpp_engine
        self.streaming_vocoder = streaming_vocoder
        self.audio_normalizer = audio_normalizer
        self.device = device
        self.vocoder_chunk_size = vocoder_chunk_size
        self.play_mode = play_mode
        self.chunk_output_dir = chunk_output_dir
        self.profile_timing = profile_timing
        self.profile_every = max(1, int(profile_every))

        # Profiling statistics (accumulated in seconds)
        self._profile = {
            'input_prepare': 0.0,
            'cpp': 0.0,
            'to_torch': 0.0,
            'normalizer': 0.0,
            'pending': 0.0,
            'concat': 0.0,
            'to_device': 0.0,
            'vocoder': 0.0,
            'audio_post': 0.0,
            'chunk_total': 0.0,
        }
        self._profile_cpp_calls = 0
        self._profile_vocoder_calls = 0

        self.emg_queue = queue.Queue(maxsize=64)
        self.processed_queue = queue.Queue(maxsize=64)

        self.audio_segments = []
        self.total_audio_samples = 0
        self.chunk_count = 0
        self.vocoder_count = 0
        self.inference_done_time = None

        # Real-time playback process
        self._rt_queue = None
        self._rt_proc = None

        if self.chunk_output_dir:
            os.makedirs(self.chunk_output_dir, exist_ok=True)

    @staticmethod
    def _now():
        """High-resolution monotonic timer."""
        return time.perf_counter()

    def _sync_device(self):
        """
        CUDA operations are asynchronous; synchronize to measure vocoder timing correctly.
        On Raspberry Pi / CPU this function has negligible overhead.
        """
        try:
            dev = self.device
            if isinstance(dev, str):
                is_cuda = dev.startswith('cuda')
            else:
                is_cuda = getattr(dev, 'type', None) == 'cuda'

            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
        except Exception:
            pass

    @staticmethod
    def _ms(seconds):
        return seconds * 1000.0

    def _log_profile_line(self, chunk_idx, audio_ms, times):
        """
         vocoder audio嚗� breakdown��
        """
        total_ms = self._ms(times['chunk_total'])
        cpp_ms = self._ms(times['cpp'])
        voc_ms = self._ms(times['vocoder'])

        if audio_ms > 0:
            total_rtf = total_ms / audio_ms
            cpp_rtf = cpp_ms / audio_ms
            voc_rtf = voc_ms / audio_ms
        else:
            total_rtf = cpp_rtf = voc_rtf = 0.0

        logging.info(
            '[PERF] '
            f'Chunk {chunk_idx:4d} | '
            f'prepare {self._ms(times["input_prepare"]):6.1f} ms | '
            f'C++ {cpp_ms:6.1f} ms | '
            f'torch {self._ms(times["to_torch"]):5.1f} ms | '
            f'norm {self._ms(times["normalizer"]):5.1f} ms | '
            f'pending {self._ms(times["pending"]):5.1f} ms | '
            f'cat {self._ms(times["concat"]):5.1f} ms | '
            f'to_dev {self._ms(times["to_device"]):5.1f} ms | '
            f'VOCODER {voc_ms:6.1f} ms | '
            f'post {self._ms(times["audio_post"]):5.1f} ms | '
            f'TOTAL {total_ms:6.1f} ms | '
            f'audio {audio_ms:6.1f} ms | '
            f'RTF total={total_rtf:.3f} '
            f'cpp={cpp_rtf:.3f} voc={voc_rtf:.3f}'
        )

    def _log_profile_summary(self):
        """Print average timings and bottleneck percentages after the run finishes."""
        if not self.profile_timing or self._profile_cpp_calls == 0:
            return

        cpp_calls = max(1, self._profile_cpp_calls)
        voc_calls = max(1, self._profile_vocoder_calls)

        cpp_total = self._profile['cpp']
        voc_total = self._profile['vocoder']
        total = self._profile['chunk_total']

        # All overhead outside C++ inference and the vocoder
        overhead = max(0.0, total - cpp_total - voc_total)

        denom = total if total > 0 else 1.0

        logging.info('=' * 88)
        logging.info('[PERF SUMMARY]')
        logging.info(
            f'  C++ CNN+Mamba : total {self._ms(cpp_total):8.1f} ms | '
            f'avg/call {self._ms(cpp_total / cpp_calls):7.1f} ms | '
            f'{100.0 * cpp_total / denom:5.1f}%'
        )
        logging.info(
            f'  Vocoder        : total {self._ms(voc_total):8.1f} ms | '
            f'avg/call {self._ms(voc_total / voc_calls):7.1f} ms | '
            f'{100.0 * voc_total / denom:5.1f}%'
        )
        logging.info(
            f'  Other overhead : total {self._ms(overhead):8.1f} ms | '
            f'{100.0 * overhead / denom:5.1f}%'
        )

        for key in (
            'input_prepare', 'to_torch', 'normalizer',
            'pending', 'concat', 'to_device', 'audio_post'
        ):
            logging.info(
                f'    {key:13s}: {self._ms(self._profile[key]):8.1f} ms'
            )

        if cpp_total >= voc_total and cpp_total >= overhead:
            bottleneck = 'C++ CNN+Mamba'
        elif voc_total >= cpp_total and voc_total >= overhead:
            bottleneck = 'PyTorch Vocoder'
        else:
            bottleneck = 'Python / data movement / audio I/O overhead'

        logging.info(f'  ==> Current largest bottleneck: {bottleneck}')
        logging.info('=' * 88)

    def start(self, emg_source):
        """Start the pipeline and initialize the C++ streaming state."""
        self.cpp_engine.init_state()

        # Start the real-time playback process
        if self.play_mode == 'realtime':
            self._rt_queue = multiprocessing.Queue()
            self._rt_proc = multiprocessing.Process(
                target=_realtime_playback_worker,
                args=(self._rt_queue, AUDIO_SR),
                daemon=True,
            )
            self._rt_proc.start()
            logging.info('Real-time playback process started (pacat)')

        self._t1 = threading.Thread(target=self._emg_capture_loop,
                                     args=(emg_source,), daemon=True)
        self._t2 = threading.Thread(target=self._preprocess_loop, daemon=True)
        self._t3 = threading.Thread(target=self._inference_loop, daemon=True)

        self._t1.start()
        self._t2.start()
        self._t3.start()

    def wait(self, wav_path=None):
        """Wait for inference to finish and handle playback according to play_mode."""
        self._t1.join()
        self._t2.join()
        self._t3.join()

        # Stop the real-time playback process
        if self._rt_queue is not None:
            self._rt_queue.put(SENTINEL_BYTES)
        if self._rt_proc is not None:
            self._rt_proc.join(timeout=30)
            if self._rt_proc.is_alive():
                self._rt_proc.kill()
            logging.info('Real-time playback stopped')

        # paplay mode: play the complete WAV after inference
        if self.play_mode == 'paplay' and wav_path and len(self.audio_segments) > 0:
            self._play_with_paplay(wav_path)

    def _send_realtime(self, audio):
        """audio process (s16le )"""
        if self._rt_queue is not None and len(audio) > 0:
            try:
                audio_i16 = np.clip(audio, -1.0, 1.0)
                audio_i16 = (audio_i16 * 32767).astype(np.int16)
                self._rt_queue.put_nowait(audio_i16.tobytes())
            except Exception:
                pass

    def _play_with_paplay(self, wav_path):
        """Play a WAV file with paplay."""
        if shutil.which('paplay') is None:
            logging.warning('paplay not found; playback is unavailable')
            return

        full_audio = self.get_full_audio()
        sf.write(wav_path, full_audio, AUDIO_SR)
        logging.info(f'Starting playback: {wav_path}')

        try:
            proc = subprocess.Popen(
                ['paplay', wav_path],
                stderr=subprocess.DEVNULL,
            )
            proc.wait()
            logging.info('Playback finished')
        except Exception as e:
            logging.warning(f'Playback error: {e}')

    def get_full_audio(self):
        if self.audio_segments:
            return np.concatenate(self.audio_segments)
        return np.array([], dtype=np.float32)

    # -- Thread 1: EMG capture -----------------------------------
    def _emg_capture_loop(self, emg_source):
        try:
            for chunk in emg_source:
                self.emg_queue.put(chunk)
        except Exception as e:
            logging.error(f'EMG capture error: {e}')
        finally:
            self.emg_queue.put(SENTINEL)

    # -- Thread 2: Preprocessing (keep data as NumPy) ------------
    def _preprocess_loop(self):
        try:
            while True:
                raw = self.emg_queue.get()
                if raw is SENTINEL:
                    break
                # Pass the NumPy array directly; no Torch conversion is needed
                self.processed_queue.put(raw)
        except Exception as e:
            logging.error(f'Preprocessing error: {e}')
        finally:
            self.processed_queue.put(SENTINEL)

    # -- Thread 3: Inference (C++ engine + PyTorch vocoder) -------
    def _inference_loop(self):
        mel_pending = []

        try:
            while True:
                emg_chunk = self.processed_queue.get()
                if emg_chunk is SENTINEL:
                    break

                self.chunk_count += 1
                chunk_t0 = self._now()

                times = {
                    'input_prepare': 0.0,
                    'cpp': 0.0,
                    'to_torch': 0.0,
                    'normalizer': 0.0,
                    'pending': 0.0,
                    'concat': 0.0,
                    'to_device': 0.0,
                    'vocoder': 0.0,
                    'audio_post': 0.0,
                    'chunk_total': 0.0,
                }

                # ------------------------------------------------------
                # 1) Python/NumPy -> pybind input preparation
                # ------------------------------------------------------
                t = self._now()
                chunk_len = emg_chunk.shape[0]
                flat_emg = emg_chunk.flatten().tolist()
                times['input_prepare'] = self._now() - t

                # ------------------------------------------------------
                # 2) C++ CNN + Mamba
                # ------------------------------------------------------
                t = self._now()
                mel_list = self.cpp_engine.forward_streaming(
                    flat_emg,
                    chunk_len
                )
                times['cpp'] = self._now() - t
                self._profile_cpp_calls += 1

                # ------------------------------------------------------
                # 3) C++ result -> Torch tensor
                # ------------------------------------------------------
                t = self._now()
                mel_frames = torch.tensor(
                    mel_list,
                    dtype=torch.float32
                )
                times['to_torch'] = self._now() - t

                # ------------------------------------------------------
                # 4) Mel inverse normalization
                # ------------------------------------------------------
                t = self._now()
                mel_frames = self.audio_normalizer.inverse(
                    mel_frames
                )
                times['normalizer'] = self._now() - t

                # ------------------------------------------------------
                # 5) Pending-mel bookkeeping
                # ------------------------------------------------------
                t = self._now()
                mel_pending.append(mel_frames)
                total_pending = sum(
                    m.shape[0] for m in mel_pending
                )
                times['pending'] = self._now() - t

                audio = None
                audio_ms = 0.0

                if total_pending >= self.vocoder_chunk_size:
                    # --------------------------------------------------
                    # 6) Concatenate mel frames
                    # --------------------------------------------------
                    t = self._now()
                    mel_cat = torch.cat(
                        mel_pending,
                        dim=0
                    )
                    mel_pending = []
                    times['concat'] = self._now() - t

                    # --------------------------------------------------
                    # 7) CPU -> vocoder device
                    # --------------------------------------------------
                    self._sync_device()
                    t = self._now()
                    mel_device = mel_cat.to(
                        self.device
                    )
                    self._sync_device()
                    times['to_device'] = self._now() - t

                    # --------------------------------------------------
                    # 8) PyTorch vocoder
                    # --------------------------------------------------
                    self._sync_device()
                    t = self._now()
                    audio = self.streaming_vocoder.synthesize_chunk(
                        mel_device
                    )
                    self._sync_device()
                    times['vocoder'] = self._now() - t
                    self._profile_vocoder_calls += 1

                    # --------------------------------------------------
                    # 9) Audio copy / playback queue / optional disk I/O
                    # --------------------------------------------------
                    t = self._now()

                    if len(audio) > 0:
                        self.audio_segments.append(
                            audio.copy()
                        )
                        self.total_audio_samples += len(audio)
                        self.vocoder_count += 1

                        audio_ms = (
                            len(audio) / AUDIO_SR * 1000.0
                        )

                        # Real-time playback only enqueues bytes for another process
                        self._send_realtime(audio)

                        # If chunk_output_dir is enabled, disk-write time is included in post
                        if self.chunk_output_dir:
                            chunk_path = os.path.join(
                                self.chunk_output_dir,
                                f'chunk_{self.vocoder_count:03d}.wav'
                            )
                            sf.write(
                                chunk_path,
                                audio,
                                AUDIO_SR
                            )

                    times['audio_post'] = self._now() - t

                times['chunk_total'] = self._now() - chunk_t0

                # Accumulate profiling statistics
                for key in self._profile:
                    self._profile[key] += times[key]

                # Log a full breakdown for chunks that invoke the vocoder
                if (
                    self.profile_timing
                    and audio is not None
                    and len(audio) > 0
                    and (self.chunk_count % self.profile_every == 0)
                ):
                    self._log_profile_line(
                        self.chunk_count,
                        audio_ms,
                        times
                    )

                # For chunks that do not invoke the vocoder, still report C++ timing
                elif (
                    self.profile_timing
                    and self.chunk_count % self.profile_every == 0
                ):
                    logging.info(
                        '[PERF] '
                        f'Chunk {self.chunk_count:4d} | '
                        f'prepare {self._ms(times["input_prepare"]):6.1f} ms | '
                        f'C++ {self._ms(times["cpp"]):6.1f} ms | '
                        f'torch {self._ms(times["to_torch"]):5.1f} ms | '
                        f'norm {self._ms(times["normalizer"]):5.1f} ms | '
                        f'TOTAL {self._ms(times["chunk_total"]):6.1f} ms | '
                        '(mel buffering; vocoder not called on this chunk)'
                    )

            # ----------------------------------------------------------
            # Flush remaining mel frames
            # ----------------------------------------------------------
            if mel_pending:
                t0 = self._now()

                mel_cat = torch.cat(
                    mel_pending,
                    dim=0
                )

                self._sync_device()
                t_voc = self._now()

                audio = self.streaming_vocoder.synthesize_chunk(
                    mel_cat.to(self.device)
                )

                self._sync_device()
                flush_voc_ms = (
                    self._now() - t_voc
                ) * 1000.0

                if len(audio) > 0:
                    self.audio_segments.append(
                        audio.copy()
                    )
                    self.total_audio_samples += len(audio)
                    self._send_realtime(audio)

                if self.profile_timing:
                    logging.info(
                        f'[PERF] Final pending mel vocoder: '
                        f'{flush_voc_ms:.1f} ms'
                    )

            # Flush the vocoder tail
            self._sync_device()
            t = self._now()
            tail = self.streaming_vocoder.flush()
            self._sync_device()
            flush_ms = (
                self._now() - t
            ) * 1000.0

            if len(tail) > 0:
                self.audio_segments.append(
                    tail.copy()
                )
                self.total_audio_samples += len(tail)
                self._send_realtime(tail)

            if self.profile_timing:
                logging.info(
                    f'[PERF] Vocoder flush(): {flush_ms:.1f} ms'
                )

        except Exception as e:
            logging.error(
                f'Inference error: {e}',
                exc_info=True
            )

        self.inference_done_time = time.time()

        self._log_profile_summary()

        logging.info(
            f'C++ inference finished '
            f'({self.chunk_count} chunks, '
            f'{self.total_audio_samples/AUDIO_SR:.2f}s audio)'
        )