"""
EMG 信號模擬器 — 模擬即時 EMG 訊號輸入

從已有的 EMG 資料集中讀取一個樣本的 raw_emg 資料，
然後以 chunk 方式逐步送出，模擬即時 EMG 信號源。

每個 chunk 包含 chunk_frames * 8 個 raw EMG samples，
因為模型 CNN 前端的 total stride = 8，
所以每 8 個 raw samples 對應 1 個 mel-rate 輸出幀。
"""

import time
import numpy as np


class EMGSimulator:
    """
    模擬即時 EMG 訊號源

    用法:
        simulator = EMGSimulator(raw_emg, chunk_frames=10)
        for chunk in simulator:
            # chunk.shape = (80, 8)  => 10 mel frames * 8 raw samples/frame
            process(chunk)
    """

    def __init__(self, raw_emg, chunk_frames=10, realtime=False,
                 sample_rate=689.06):
        """
        Args:
            raw_emg: numpy array, shape (T*8, 8), 已預處理並正規化的原始 EMG
            chunk_frames: 每個 chunk 對應多少 mel 幀 (每幀 = 8 個 raw samples)
            realtime: 是否模擬真實時間延遲
            sample_rate: raw EMG 取樣率 (Hz)
        """
        self.raw_emg = raw_emg
        self.chunk_size = chunk_frames * 8
        self.realtime = realtime
        self.sample_rate = sample_rate
        self.total_raw_samples = len(raw_emg)
        self.total_frames = self.total_raw_samples // 8

    def __iter__(self):
        """逐 chunk 送出 EMG 資料，realtime 模式會對齊真實時間"""
        idx = 0
        start_wall = time.perf_counter()

        while idx < self.total_raw_samples:
            end = min(idx + self.chunk_size, self.total_raw_samples)
            chunk_len = ((end - idx) // 8) * 8
            if chunk_len == 0:
                break

            chunk = self.raw_emg[idx:idx + chunk_len]
            next_idx = idx + chunk_len

            if self.realtime:
                # 這個 chunk 結束的「真實時間」
                target_time = start_wall + next_idx / self.sample_rate
                now = time.perf_counter()
                wait = target_time - now
                if wait > 0:
                    time.sleep(wait)

            yield chunk
            idx = next_idx

    @property
    def total_chunks(self):
        usable = (self.total_raw_samples // 8) * 8
        return max(1, (usable + self.chunk_size - 1) // self.chunk_size)

    @property
    def duration_seconds(self):
        return self.total_raw_samples / self.sample_rate
