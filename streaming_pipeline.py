"""
多執行緒即時 EMG → 語音 Pipeline

架構:
    Thread 1 (EMG 擷取)    → emg_queue
    Thread 2 (前處理)       → processed_queue
    Thread 3 (推論引擎)     → audio_segments[]

    播放模式:
      - 'paplay': 推論完成後用 paplay 播 WAV (乾淨, 有延遲)
      - 'realtime': 推論中即時用 pacat 串流播放 (可能有雜音)
      - 'none': 不播放
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


class StreamingPipeline:
    def __init__(self, model, streaming_vocoder, audio_normalizer,
                 device, vocoder_chunk_size=10, play_mode='paplay',
                 chunk_output_dir=None,
                 on_emg_chunk=None, on_audio_chunk=None,
                 on_mel_chunk=None, on_progress=None):
        """
        Args:
            play_mode: 'paplay' | 'realtime' | 'none'
            on_emg_chunk: callback(raw_emg_np) — called when an EMG chunk enters
            on_audio_chunk: callback(audio_np) — called when vocoder produces audio
            on_mel_chunk: callback(mel_np) — called when model produces mel frames
            on_progress: callback(chunk_idx, total_chunks) — for progress tracking
        """
        self.model = model
        self.streaming_vocoder = streaming_vocoder
        self.audio_normalizer = audio_normalizer
        self.device = device
        self.vocoder_chunk_size = vocoder_chunk_size
        self.play_mode = play_mode
        self.chunk_output_dir = chunk_output_dir

        # Optional callbacks for UI / observers
        self.on_emg_chunk = on_emg_chunk
        self.on_audio_chunk = on_audio_chunk
        self.on_mel_chunk = on_mel_chunk
        self.on_progress = on_progress

        self.emg_queue = queue.Queue(maxsize=64)
        self.processed_queue = queue.Queue(maxsize=64)

        self.model_state = None
        self.audio_segments = []
        self.total_audio_samples = 0
        self.chunk_count = 0
        self.total_chunks = 0
        self.vocoder_count = 0
        self.inference_done_time = None

        # realtime 播放用
        self._rt_queue = None
        self._rt_proc = None

        if self.chunk_output_dir:
            os.makedirs(self.chunk_output_dir, exist_ok=True)

    def start(self, emg_source):
        self.model.eval()
        self.model_state = self.model.init_streaming_state(self.device)

        # Store total_chunks for progress tracking
        if hasattr(emg_source, 'total_chunks'):
            self.total_chunks = emg_source.total_chunks

        # 啟動即時播放 process
        if self.play_mode == 'realtime':
            self._rt_queue = multiprocessing.Queue()
            self._rt_proc = multiprocessing.Process(
                target=_realtime_playback_worker,
                args=(self._rt_queue, AUDIO_SR),
                daemon=True,
            )
            self._rt_proc.start()
            logging.info('')

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
            logging.info('')

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
        logging.info(f'')

        try:
            proc = subprocess.Popen(
                ['paplay', wav_path],
                stderr=subprocess.DEVNULL,
            )
            proc.wait()
            logging.info('')
        except Exception as e:
            logging.warning(f'播放錯誤: {e}')

    def get_full_audio(self):
        if self.audio_segments:
            return np.concatenate(self.audio_segments)
        return np.array([], dtype=np.float32)

    # ── Thread 1 ─────────────────────────────────────────────
    def _emg_capture_loop(self, emg_source):
        try:
            for chunk in emg_source:
                self.emg_queue.put(chunk)
                if self.on_emg_chunk is not None:
                    try:
                        self.on_emg_chunk(chunk)
                    except Exception:
                        pass
        except Exception as e:
            logging.error(f'EMG 擷取錯誤: {e}')
        finally:
            self.emg_queue.put(SENTINEL)

    # ── Thread 2 ─────────────────────────────────────────────
    def _preprocess_loop(self):
        try:
            while True:
                raw = self.emg_queue.get()
                if raw is SENTINEL:
                    break
                chunk_t = torch.from_numpy(raw).float().to(self.device)
                chunk_t = chunk_t.unsqueeze(0)
                self.processed_queue.put(chunk_t)
        except Exception as e:
            logging.error(f'前處理錯誤: {e}')
        finally:
            self.processed_queue.put(SENTINEL)

    # ── Thread 3 ─────────────────────────────────────────────
    def _inference_loop(self):
        mel_pending = []

        try:
            while True:
                emg_chunk = self.processed_queue.get()
                if emg_chunk is SENTINEL:
                    break

                self.chunk_count += 1
                t0 = time.time()

                with torch.no_grad():
                    mel, aux, self.model_state = self.model.forward_streaming(
                        emg_chunk, self.model_state
                    )

                mel_frames = mel.squeeze(0)
                mel_frames = self.audio_normalizer.inverse(mel_frames.cpu())
                mel_pending.append(mel_frames)

                if self.on_mel_chunk is not None:
                    try:
                        self.on_mel_chunk(mel_frames.numpy())
                    except Exception:
                        pass
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

                        # UI callback
                        if self.on_audio_chunk is not None:
                            try:
                                self.on_audio_chunk(audio)
                            except Exception:
                                pass

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
                            f'delay {dt*1000:6.1f}ms | '
                            f'{audio_ms:6.1f}ms | '
                            f'total {self.total_audio_samples/AUDIO_SR:.2f}s'
                        )

                # Progress callback
                if self.on_progress is not None:
                    try:
                        self.on_progress(self.chunk_count, self.total_chunks)
                    except Exception:
                        pass

            # Flush
            if mel_pending:
                mel_cat = torch.cat(mel_pending, dim=0)
                audio = self.streaming_vocoder.synthesize_chunk(
                    mel_cat.to(self.device)
                )
                if len(audio) > 0:
                    self.audio_segments.append(audio.copy())
                    self.total_audio_samples += len(audio)
                    self._send_realtime(audio)
                    if self.on_audio_chunk is not None:
                        try:
                            self.on_audio_chunk(audio)
                        except Exception:
                            pass

            tail = self.streaming_vocoder.flush()
            if len(tail) > 0:
                self.audio_segments.append(tail.copy())
                self.total_audio_samples += len(tail)
                self._send_realtime(tail)
                if self.on_audio_chunk is not None:
                    try:
                        self.on_audio_chunk(tail)
                    except Exception:
                        pass

        except Exception as e:
            logging.error(f'推論錯誤: {e}', exc_info=True)

        self.inference_done_time = time.time()
