"""
EMG 通道品質分析模組 (Channel Quality Analyzer) v2

偵測 EMG 電極配戴狀況是否正確、接觸是否良好。
核心改進：不僅依賴校準期基線，更加入「絕對品質檢查」，
即使一開始就沒戴好，也能偵測出問題。

═══════════════════════════════════════════════════════════════════
雙層評估架構：
═══════════════════════════════════════════════════════════════════

  Layer 1 — 絕對品質檢查 (Absolute Quality Check)
    不需要校準，基於 EMG 訊號的物理特性判斷：
    ① 訊號存在性：RMS > 絕對最低門檻？std > 0？
    ② 波形合理性：峰度 (kurtosis) 是否接近高斯分布？
    ③ 頻譜特性：  過零率 (ZCR) 是否符合 EMG 頻寬？
    ④ 通道獨立性：是否與其他通道過度相關？(共模干擾偵測)
    → 即使校準時就戴不好，這層也能偵測出問題

  Layer 2 — 相對品質檢查 (Relative Quality Check)
    基於校準期基線的比較（僅在校準通過後啟用）：
    ① 振幅一致性：即時 RMS vs 校準 RMS
    ② 訊號穩定性：RMS 變異係數 vs 校準 CV
    ③ 偽影汙染度：超過 ±4σ 的 spike 比例
    ④ 基線品質：  DC 偏移漂移量
    → 偵測「原本好但後來變差」的狀況

  最終分數 = min(絕對分數, 相對分數)
    → 任何一層偵測到問題就扣分，不會被壞的校準遮蔽

═══════════════════════════════════════════════════════════════════
校準驗證與自動重新校準：
═══════════════════════════════════════════════════════════════════

  - 校準期間同時執行絕對品質檢查
  - 僅當絕對品質 ≥ 門檻時才接受校準資料
  - 若校準期間品質不佳 → 延長校準 / 顯示警告
  - 運行中若電極被重新調整 → 自動偵測並觸發重新校準

═══════════════════════════════════════════════════════════════════
信號格數對照表：
═══════════════════════════════════════════════════════════════════
  85-100 分 → 4 bars (優秀) — 電極接觸完美，訊號品質極佳
  65-84  分 → 3 bars (良好) — 接觸良好，適合使用
  45-64  分 → 2 bars (普通) — 接觸尚可，可能有輕微問題
  25-44  分 → 1 bar  (差)   — 接觸不良，建議重新調整電極
   0-24  分 → 0 bars (極差) — 斷線或嚴重接觸問題

═══════════════════════════════════════════════════════════════════
使用方式：
═══════════════════════════════════════════════════════════════════
    from channel_quality import ChannelQualityAnalyzer, ChannelStatus

    analyzer = ChannelQualityAnalyzer(
        num_channels=8,
        sample_rate=689.06,
        calibration_duration=3.0
    )
    statuses = analyzer.analyze_chunk(emg_chunk)  # shape=(num_channels, N)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging


# ═══════════════════════════════════════════════════════════════════
# 資料類別
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ChannelStatus:
    """單一通道的品質狀態"""
    channel_id: int             # 通道編號 (0-based)
    bars: int                   # 信號格數 0-4
    quality: str                # 品質文字: excellent/good/fair/poor/bad/uncalibrated
    score: float                # 品質分數 0-100
    detail: str                 # 人類可讀的詳細描述
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class _ChannelCalibration:
    """單一通道的校準基線數據"""
    valid: bool = False         # 此通道的校準是否有效
    mean: float = 0.0
    std: float = 1e-6
    rms: float = 1e-6
    cv: float = 0.0
    p5: float = 0.0
    p95: float = 0.0
    zcr: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# EMG 訊號物理特性常數
# ═══════════════════════════════════════════════════════════════════
# 以下閾值基於經過前處理的 EMG 訊號特性：
#   前處理流程: raw * emg_scale → notch → drift_removal → /20 → tanh(x/50)*50
#
# 正常的表面 EMG 經過此流程後應具備以下特性：
#   - RMS: 大約 0.01 ~ 5.0 (取決於肌肉活動程度)
#   - 峰度 (kurtosis): 接近 3 (高斯分布)，通常 1.8 ~ 8.0
#   - 過零率 (ZCR): EMG 頻寬 ~20-300 Hz，在 689 Hz 取樣率下 ZCR 約 0.1 ~ 0.7
#   - 各通道應具有一定的獨立性 (相關係數 < 0.95)

_ABSOLUTE_THRESHOLDS = {
    # 死通道偵測
    'flatline_std': 1e-5,          # std 低於此 → 完全無訊號變化
    'rms_floor': 0.005,            # RMS 低於此 → 幾乎無接觸

    # 波形合理性 (峰度 kurtosis)
    # 高斯分布 kurtosis=3, EMG 通常 1.8~8.0
    # 過低 → 太均勻 (不像真實 EMG)
    # 過高 → spike 主導 (動作偽影或電極碰撞)
    'kurtosis_min': 1.5,
    'kurtosis_max': 15.0,
    'kurtosis_ideal_low': 2.0,     # 理想範圍下限
    'kurtosis_ideal_high': 6.0,    # 理想範圍上限

    # 頻譜特性 (過零率 ZCR)
    # EMG 有寬頻內容，ZCR 不會太低 (純 DC) 也不會太高 (純高頻雜訊)
    # 在 689 Hz 取樣率下：
    #   DC/無訊號:  ZCR ≈ 0
    #   50Hz 工頻:  ZCR ≈ 0.14
    #   正常 EMG:   ZCR ≈ 0.15 ~ 0.65
    #   白噪音:     ZCR ≈ 0.50
    'zcr_min': 0.05,               # 低於此 → 可能是 DC 偏移或無訊號
    'zcr_max': 0.85,               # 高於此 → 可能是高頻雜訊
    'zcr_ideal_low': 0.12,
    'zcr_ideal_high': 0.65,

    # 通道獨立性 (cross-correlation)
    # 若兩通道相關係數 > 此值 → 可能是共模干擾
    'max_correlation': 0.92,

    # 動態範圍 (Dynamic Range): std / RMS 比值
    # 正常 EMG 是零均值隨機訊號 → std ≈ RMS → 比值接近 1.0
    # 純 DC 偏移 + 微小雜訊       → std ≪ RMS → 比值接近 0
    # 這能抓到「有大 DC 偏移但幾乎沒有真實 EMG 活動」的情況
    # (高阻抗接觸 / 電極浮接 / 半波整流偽影)
    'dynamic_range_min': 0.1,      # 低於此 → DC 主導 (嚴重)
    'dynamic_range_ideal': 0.5,    # 高於此 → 正常 EMG
}


# ═══════════════════════════════════════════════════════════════════
# 主分析器
# ═══════════════════════════════════════════════════════════════════

class ChannelQualityAnalyzer:
    """
    EMG 通道品質即時分析器 (雙層架構)。

    參數：
      num_channels        : EMG 通道數 (預設 8)
      sample_rate         : 取樣率 Hz (預設 689.06)
      calibration_duration: 校準期長度 秒 (預設 3.0)
      ema_alpha           : 分數平滑係數 (0-1, 越小越平滑)
    """

    _SCORE_THRESHOLDS = [
        (85, 4, 'excellent', '優秀 — 電極接觸完美'),
        (65, 3, 'good',      '良好 — 適合使用'),
        (45, 2, 'fair',      '普通 — 可能有輕微問題'),
        (25, 1, 'poor',      '差 — 建議調整電極'),
        (0,  0, 'bad',       '極差 — 斷線或嚴重接觸問題'),
    ]

    # 校準驗證：絕對分數必須達到此門檻才接受校準資料
    CALIB_ACCEPT_THRESHOLD = 40.0

    # 自動重新校準：若絕對分數持續高於此值 N 秒，觸發重新校準
    RECALIB_ABS_THRESHOLD = 70.0
    RECALIB_DURATION = 3.0  # 需要持續好幾秒才觸發

    def __init__(
        self,
        num_channels: int = 8,
        sample_rate: float = 689.06,
        calibration_duration: float = 3.0,
        ema_alpha: float = 0.3,
    ):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.calibration_samples = int(sample_rate * calibration_duration)
        self.ema_alpha = ema_alpha

        # ── 校準狀態 ──────────────────────────────────────────
        self._calibrated = False
        self._calib_buffers: List[List[np.ndarray]] = [[] for _ in range(num_channels)]
        self._calib_good_buffers: List[List[np.ndarray]] = [[] for _ in range(num_channels)]
        self._calib_good_count: List[int] = [0] * num_channels
        self._calib_total_count: int = 0
        self._calibrations: List[_ChannelCalibration] = [
            _ChannelCalibration() for _ in range(num_channels)
        ]

        # ── 分析狀態 ──────────────────────────────────────────
        self._smoothed_scores: np.ndarray = np.full(num_channels, 50.0)
        self._rms_window_size = max(1, int(sample_rate / 34))
        self._rms_history: List[List[float]] = [[] for _ in range(num_channels)]

        # ── 自動重新校準偵測 ──────────────────────────────────
        # 紀錄每通道連續多少樣本的絕對品質是好的
        self._recalib_good_streak: List[int] = [0] * num_channels
        self._recalib_samples_needed = int(sample_rate * self.RECALIB_DURATION)
        self._recalib_buffers: List[List[np.ndarray]] = [[] for _ in range(num_channels)]

        # ── 通道間相關性分析的累積緩衝 ────────────────────────
        self._corr_buffer: List[np.ndarray] = []
        self._corr_buffer_size = int(sample_rate * 0.5)  # ~0.5 秒
        self._corr_buffer_count = 0
        self._channel_correlations: np.ndarray = np.zeros((num_channels, num_channels))

    # ═══════════════════════════════════════════════════════════
    # 公開 API
    # ═══════════════════════════════════════════════════════════

    def analyze_chunk(self, chunk: np.ndarray) -> List[ChannelStatus]:
        """
        分析一個 EMG chunk 的通道品質。

        參數：
          chunk: shape=(num_channels, chunk_size), 已前處理的 EMG

        回傳：
          List[ChannelStatus]
        """
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        n_ch = min(chunk.shape[0], self.num_channels)
        chunk_size = chunk.shape[1]

        # 更新通道間相關性
        self._update_correlation(chunk[:n_ch])

        # ── Step 1: 對每個通道計算絕對品質分數 ────────────────
        abs_scores = []
        abs_details = []
        for ch in range(n_ch):
            score, detail = self._compute_absolute_score(chunk[ch], ch)
            abs_scores.append(score)
            abs_details.append(detail)

        # ── Step 2: 校準階段處理 ──────────────────────────────
        if not self._calibrated:
            return self._handle_calibration_phase(
                chunk, n_ch, chunk_size, abs_scores, abs_details
            )

        # ── Step 3: 已校準 — 結合絕對 + 相對分數 ─────────────
        statuses = []
        for ch in range(n_ch):
            signal = chunk[ch]
            calib = self._calibrations[ch]
            abs_score = abs_scores[ch]
            abs_detail = abs_details[ch]

            if calib.valid:
                # 相對品質分數
                rel_score, rel_detail = self._compute_relative_score(signal, calib, ch)

                # 最終分數 = min(絕對, 相對)
                # → 任何一層偵測到問題就會反映
                raw_score = min(abs_score, rel_score)

                detail_lines = [
                    f'絕對品質: {abs_score:.0f}/100 — {abs_detail}',
                    f'相對品質: {rel_score:.0f}/100 — {rel_detail}',
                ]
            else:
                # 此通道未成功校準 → 僅用絕對分數
                raw_score = abs_score
                detail_lines = [
                    f'絕對品質: {abs_score:.0f}/100 — {abs_detail}',
                    '(此通道未成功校準，僅使用絕對品質評估)',
                ]

            # EMA 平滑
            self._smoothed_scores[ch] = (
                self.ema_alpha * raw_score
                + (1 - self.ema_alpha) * self._smoothed_scores[ch]
            )
            final_score = self._smoothed_scores[ch]

            bars, quality, quality_desc = self._score_to_bars(final_score)
            detail_lines.insert(0, f'總分: {final_score:.0f}/100 ({quality_desc})')

            statuses.append(ChannelStatus(
                channel_id=ch,
                bars=bars,
                quality=quality,
                score=final_score,
                detail='\n'.join(detail_lines),
                metrics={
                    'absolute_score': abs_score,
                    'relative_score': rel_score if calib.valid else -1,
                    'raw_score': raw_score,
                    'smoothed_score': final_score,
                },
            ))

            # ── 自動重新校準偵測 ──────────────────────────────
            self._check_recalibration(ch, signal, abs_score, calib)

        return statuses

    def reset(self):
        """重設分析器，回到校準狀態。"""
        self.__init__(
            self.num_channels, self.sample_rate,
            self.calibration_samples / self.sample_rate,
            self.ema_alpha,
        )

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ═══════════════════════════════════════════════════════════
    # Layer 1: 絕對品質檢查 (不需要校準)
    # ═══════════════════════════════════════════════════════════

    def _compute_absolute_score(self, signal: np.ndarray, ch: int) -> tuple:
        """
        基於 EMG 訊號的物理特性，計算不依賴校準的品質分數 (0-100)。

        子維度 (各 0-20 分)：
          ① 訊號存在性：是否有真實訊號？
          ② 波形合理性：峰度是否接近高斯？
          ③ 頻譜特性：  過零率是否符合 EMG？
          ④ 通道獨立性：是否受共模干擾？
          ⑤ 動態範圍：  std/RMS 比值，偵測 DC 主導訊號
        """
        T = _ABSOLUTE_THRESHOLDS
        details = []

        # ── ① 訊號存在性 (0-20) ─────────────────────────────
        std = float(np.std(signal))
        rms = float(np.sqrt(np.mean(signal ** 2)))

        if std < T['flatline_std']:
            s1 = 0.0
            details.append('死通道')
        elif rms < T['rms_floor']:
            s1 = 4.0 * (rms / T['rms_floor'])
            details.append(f'訊號微弱(RMS={rms:.4f})')
        else:
            s1 = 20.0
            details.append('訊號正常')

        # 如果死通道，後續檢查無意義
        if s1 == 0:
            return 0.0, '死通道 (無訊號)'

        # ── ② 波形合理性: 峰度 (0-20) ───────────────────────
        if len(signal) >= 4 and std > T['flatline_std']:
            kurt = float(self._kurtosis(signal))
        else:
            kurt = 3.0

        if kurt < T['kurtosis_min']:
            s2 = 4.0
            details.append(f'波形過於均勻(K={kurt:.1f})')
        elif kurt > T['kurtosis_max']:
            s2 = 4.0
            details.append(f'spike嚴重(K={kurt:.1f})')
        elif T['kurtosis_ideal_low'] <= kurt <= T['kurtosis_ideal_high']:
            s2 = 20.0
            details.append(f'波形正常(K={kurt:.1f})')
        else:
            if kurt < T['kurtosis_ideal_low']:
                ratio = (kurt - T['kurtosis_min']) / (T['kurtosis_ideal_low'] - T['kurtosis_min'])
            else:
                ratio = (T['kurtosis_max'] - kurt) / (T['kurtosis_max'] - T['kurtosis_ideal_high'])
            s2 = 4.0 + 16.0 * max(0, min(1, ratio))
            details.append(f'波形略偏(K={kurt:.1f})')

        # ── ③ 頻譜特性: 過零率 ZCR (0-20) ───────────────────
        mean_val = float(np.mean(signal))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal - mean_val))) > 0)
        zcr = float(zero_crossings / max(1, len(signal) - 1))

        if zcr < T['zcr_min']:
            s3 = 4.0
            details.append(f'疑似DC/無活動(ZCR={zcr:.2f})')
        elif zcr > T['zcr_max']:
            s3 = 4.0
            details.append(f'高頻雜訊主導(ZCR={zcr:.2f})')
        elif T['zcr_ideal_low'] <= zcr <= T['zcr_ideal_high']:
            s3 = 20.0
            details.append(f'頻譜正常(ZCR={zcr:.2f})')
        else:
            if zcr < T['zcr_ideal_low']:
                ratio = (zcr - T['zcr_min']) / (T['zcr_ideal_low'] - T['zcr_min'])
            else:
                ratio = (T['zcr_max'] - zcr) / (T['zcr_max'] - T['zcr_ideal_high'])
            s3 = 4.0 + 16.0 * max(0, min(1, ratio))
            details.append(f'頻譜略偏(ZCR={zcr:.2f})')

        # ── ④ 通道獨立性 (0-20) ──────────────────────────────
        max_corr = self._get_max_correlation(ch)
        if max_corr < 0:
            s4 = 16.0
            details.append('通道獨立性(資料不足)')
        elif max_corr > T['max_correlation']:
            excess = (max_corr - T['max_correlation']) / (1.0 - T['max_correlation'])
            s4 = 20.0 * max(0, 1.0 - excess * 2)
            details.append(f'疑似共模干擾(r={max_corr:.2f})')
        else:
            s4 = 20.0
            details.append(f'通道獨立(r={max_corr:.2f})')

        # ── ⑤ 動態範圍: std/RMS 比值 (0-20) ─────────────────
        # 正常 EMG: 零均值隨機訊號 → std ≈ RMS → ratio ≈ 1.0
        # DC 主導:  大偏移 + 微小變化 → std ≪ RMS → ratio → 0
        # 這能抓到電極浮接、高阻抗等產生大 DC 偏移的問題
        dynamic_ratio = std / (rms + 1e-10)

        if dynamic_ratio < T['dynamic_range_min']:
            s5 = 0.0
            details.append(f'DC主導(std/RMS={dynamic_ratio:.3f})')
        elif dynamic_ratio >= T['dynamic_range_ideal']:
            s5 = 20.0
            details.append(f'動態範圍正常(std/RMS={dynamic_ratio:.2f})')
        else:
            # 線性插值: min → 0, ideal → 20
            ratio = (dynamic_ratio - T['dynamic_range_min']) / (
                T['dynamic_range_ideal'] - T['dynamic_range_min']
            )
            s5 = 20.0 * max(0, min(1, ratio))
            details.append(f'動態範圍偏低(std/RMS={dynamic_ratio:.3f})')

        total = s1 + s2 + s3 + s4 + s5

        # 關鍵維度懲罰：若訊號存在性或動態範圍為 0 → 上限 20 分
        # 確保死通道和 DC 主導的資料不會通過校準驗證門檻
        if s1 == 0 or s5 == 0:
            total = min(total, 20.0)

        summary = ', '.join(details)
        return total, summary

    # ═══════════════════════════════════════════════════════════
    # Layer 2: 相對品質檢查 (需要校準)
    # ═══════════════════════════════════════════════════════════

    def _compute_relative_score(
        self, signal: np.ndarray, calib: _ChannelCalibration, ch: int
    ) -> tuple:
        """
        基於校準期基線的品質評估 (0-100)。
        偵測「原本好但後來變差」的狀況。

        子維度 (各 0-25 分)：
          ① 振幅一致性
          ② 訊號穩定性
          ③ 偽影汙染度
          ④ 基線品質
        """
        details = []

        # ── ① 振幅一致性 (0-25) ──────────────────────────────
        rms = float(np.sqrt(np.mean(signal ** 2)))
        ratio = rms / calib.rms

        if ratio < 0.2:
            s1 = 25.0 * (ratio / 0.2)
            details.append(f'振幅過低({ratio:.1f}x)')
        elif ratio > 5.0:
            excess = (ratio - 5.0) / 5.0
            s1 = 25.0 * max(0, 1.0 - excess)
            details.append(f'振幅過高({ratio:.1f}x)')
        elif 0.5 <= ratio <= 2.0:
            s1 = 25.0
            details.append(f'振幅正常({ratio:.2f}x)')
        else:
            # 0.2~0.5 或 2.0~5.0
            if ratio < 0.5:
                s1 = 15.0 + 10.0 * (ratio - 0.2) / 0.3
            else:
                s1 = 15.0 + 10.0 * (5.0 - ratio) / 3.0
            details.append(f'振幅偏離({ratio:.2f}x)')

        # ── ② 訊號穩定性 (0-25) ──────────────────────────────
        chunk_rms = rms
        self._rms_history[ch].append(chunk_rms)
        if len(self._rms_history[ch]) > self._rms_window_size:
            self._rms_history[ch] = self._rms_history[ch][-self._rms_window_size:]

        if len(self._rms_history[ch]) >= 3:
            hist = np.array(self._rms_history[ch])
            hist_cv = float(np.std(hist) / (np.mean(hist) + 1e-10))
        else:
            hist_cv = 0.0

        cv_good = max(calib.cv * 1.5, 0.1)
        cv_bad = max(calib.cv * 5.0, 0.5)

        if hist_cv <= cv_good:
            s2 = 25.0
            details.append(f'穩定(CV={hist_cv:.3f})')
        elif hist_cv >= cv_bad:
            s2 = 0.0
            details.append(f'不穩定(CV={hist_cv:.3f})')
        else:
            r = (hist_cv - cv_good) / (cv_bad - cv_good)
            s2 = 25.0 * (1.0 - r)
            details.append(f'穩定性中等(CV={hist_cv:.3f})')

        # ── ③ 偽影汙染度 (0-25) ──────────────────────────────
        threshold_hi = calib.mean + 4.0 * calib.std
        threshold_lo = calib.mean - 4.0 * calib.std
        n_spikes = int(np.sum((signal > threshold_hi) | (signal < threshold_lo)))
        spike_ratio = n_spikes / max(1, len(signal))

        if spike_ratio <= 0.001:
            s3 = 25.0
            details.append(f'無偽影')
        elif spike_ratio >= 0.05:
            s3 = 0.0
            details.append(f'偽影嚴重({spike_ratio:.1%})')
        else:
            s3 = 25.0 * (1.0 - spike_ratio / 0.05)
            details.append(f'輕微偽影({spike_ratio:.1%})')

        # ── ④ 基線品質 (0-25) ────────────────────────────────
        current_mean = float(np.mean(signal))
        drift_sigma = abs(current_mean - calib.mean) / calib.std

        if drift_sigma <= 0.5:
            s4 = 25.0
            details.append(f'基線穩定({drift_sigma:.1f}s)')
        elif drift_sigma >= 3.0:
            s4 = 0.0
            details.append(f'基線漂移({drift_sigma:.1f}s)')
        else:
            r = (drift_sigma - 0.5) / 2.5
            s4 = 25.0 * (1.0 - r)
            details.append(f'基線略移({drift_sigma:.1f}s)')

        total = s1 + s2 + s3 + s4
        summary = ', '.join(details)
        return total, summary

    # ═══════════════════════════════════════════════════════════
    # 校準流程 (含驗證)
    # ═══════════════════════════════════════════════════════════

    def _handle_calibration_phase(
        self, chunk, n_ch, chunk_size, abs_scores, abs_details
    ) -> List[ChannelStatus]:
        """
        校準階段處理。

        關鍵設計：只有絕對品質達標的資料才會被納入校準基線。
        這樣即使一開始沒戴好：
          - 壞的資料不會污染校準基線
          - 使用者可以看到即時的絕對品質分數來調整電極
          - 戴好之後的資料才會被收集，校準才會完成
        """
        self._calib_total_count += chunk_size

        statuses = []
        all_calibrated = True

        for ch in range(n_ch):
            abs_score = abs_scores[ch]

            # 判斷這個 chunk 對此通道是否是好的校準資料
            is_good = abs_score >= self.CALIB_ACCEPT_THRESHOLD

            if is_good:
                self._calib_good_buffers[ch].append(chunk[ch].copy())
                self._calib_good_count[ch] += chunk_size

            # 此通道是否已收集足夠好的校準資料
            ch_calibrated = self._calib_good_count[ch] >= self.calibration_samples
            if not ch_calibrated:
                all_calibrated = False

            # EMA 平滑絕對分數
            self._smoothed_scores[ch] = (
                self.ema_alpha * abs_score
                + (1 - self.ema_alpha) * self._smoothed_scores[ch]
            )
            final_score = self._smoothed_scores[ch]
            bars, quality_str, quality_desc = self._score_to_bars(final_score)

            # 狀態描述
            if is_good:
                good_pct = self._calib_good_count[ch] / self.calibration_samples
                if ch_calibrated:
                    detail = f'校準完成! 絕對品質: {abs_score:.0f} — {abs_details[ch]}'
                else:
                    detail = (
                        f'校準中 ({good_pct:.0%})... '
                        f'絕對品質: {abs_score:.0f} — {abs_details[ch]}'
                    )
            else:
                detail = (
                    f'品質不足，校準暫停 '
                    f'(需 >= {self.CALIB_ACCEPT_THRESHOLD:.0f}分) '
                    f'絕對品質: {abs_score:.0f} — {abs_details[ch]}'
                )
                # 不是 uncalibrated，而是顯示真實的（差）分數
                quality_str = quality_str if abs_score < 45 else quality_str

            statuses.append(ChannelStatus(
                channel_id=ch,
                bars=bars,
                quality=quality_str if not ch_calibrated else quality_str,
                score=final_score,
                detail=detail,
                metrics={
                    'absolute_score': abs_score,
                    'calibration_progress': min(1.0, self._calib_good_count[ch] / self.calibration_samples),
                    'calibration_accepted': is_good,
                },
            ))

        # 所有通道都校準完成 → 建立基線
        if all_calibrated:
            self._finalize_calibration()
            logging.info('[ChannelQuality] 所有通道校準完成!')

        return statuses

    def _finalize_calibration(self):
        """將通過品質驗證的校準資料彙整為基線統計。"""
        for ch in range(self.num_channels):
            buf = self._calib_good_buffers[ch]
            if not buf:
                self._calibrations[ch] = _ChannelCalibration(valid=False)
                logging.warning(f'[ChannelQuality] Ch{ch}: 校準失敗，無足夠好品質資料')
                continue

            data = np.concatenate(buf)

            mean_val = float(np.mean(data))
            std_val = float(np.std(data))
            rms_val = float(np.sqrt(np.mean(data ** 2)))

            # 子窗口 CV
            window_size = max(1, len(data) // 20)
            n_windows = max(1, len(data) // window_size)
            sub_rms = []
            for i in range(n_windows):
                s, e = i * window_size, min((i + 1) * window_size, len(data))
                if e > s:
                    sub_rms.append(np.sqrt(np.mean(data[s:e] ** 2)))
            sub_rms = np.array(sub_rms) if sub_rms else np.array([rms_val])
            cv = float(np.std(sub_rms) / (np.mean(sub_rms) + 1e-10))

            p5 = float(np.percentile(data, 5))
            p95 = float(np.percentile(data, 95))

            zero_crossings = np.sum(np.abs(np.diff(np.sign(data - mean_val))) > 0)
            zcr = float(zero_crossings / max(1, len(data) - 1))

            self._calibrations[ch] = _ChannelCalibration(
                valid=True,
                mean=mean_val,
                std=max(std_val, 1e-8),
                rms=max(rms_val, 1e-8),
                cv=cv, p5=p5, p95=p95, zcr=zcr,
            )

        self._calib_good_buffers = [[] for _ in range(self.num_channels)]
        self._calib_buffers = [[] for _ in range(self.num_channels)]
        self._calibrated = True

    # ═══════════════════════════════════════════════════════════
    # 自動重新校準
    # ═══════════════════════════════════════════════════════════

    def _check_recalibration(
        self, ch: int, signal: np.ndarray, abs_score: float,
        calib: _ChannelCalibration
    ):
        """
        偵測是否應該自動重新校準此通道。

        觸發條件：
          - 相對分數持續偏低（表示訊號特徵已改變）
          - 但絕對分數持續良好（表示電極接觸其實沒問題）
          - 此狀態持續超過 RECALIB_DURATION 秒
          → 可能是使用者重新調整了電極位置

        觸發時：用最近的好資料重新建立此通道的校準基線。
        """
        if not calib.valid:
            return

        rel_score, _ = self._compute_relative_score(signal, calib, ch)

        # 條件: 絕對品質好 但 相對品質差 → 可能需要重新校準
        needs_recalib = (abs_score >= self.RECALIB_ABS_THRESHOLD and rel_score < 50)

        if needs_recalib:
            self._recalib_good_streak[ch] += len(signal)
            self._recalib_buffers[ch].append(signal.copy())

            if self._recalib_good_streak[ch] >= self._recalib_samples_needed:
                # 觸發此通道的重新校準
                data = np.concatenate(self._recalib_buffers[ch])
                self._recalibrate_channel(ch, data)
                self._recalib_good_streak[ch] = 0
                self._recalib_buffers[ch] = []
                self._smoothed_scores[ch] = 75.0  # 重設平滑分數
                logging.info(
                    f'[ChannelQuality] Ch{ch}: 自動重新校準完成 '
                    f'(偵測到電極位置變更)'
                )
        else:
            self._recalib_good_streak[ch] = 0
            self._recalib_buffers[ch] = []

    def _recalibrate_channel(self, ch: int, data: np.ndarray):
        """用新資料重新校準單一通道。"""
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        rms_val = float(np.sqrt(np.mean(data ** 2)))

        window_size = max(1, len(data) // 20)
        n_windows = max(1, len(data) // window_size)
        sub_rms = []
        for i in range(n_windows):
            s, e = i * window_size, min((i + 1) * window_size, len(data))
            if e > s:
                sub_rms.append(np.sqrt(np.mean(data[s:e] ** 2)))
        sub_rms = np.array(sub_rms) if sub_rms else np.array([rms_val])
        cv = float(np.std(sub_rms) / (np.mean(sub_rms) + 1e-10))

        p5 = float(np.percentile(data, 5))
        p95 = float(np.percentile(data, 95))

        zero_crossings = np.sum(np.abs(np.diff(np.sign(data - mean_val))) > 0)
        zcr = float(zero_crossings / max(1, len(data) - 1))

        self._calibrations[ch] = _ChannelCalibration(
            valid=True,
            mean=mean_val, std=max(std_val, 1e-8), rms=max(rms_val, 1e-8),
            cv=cv, p5=p5, p95=p95, zcr=zcr,
        )
        self._rms_history[ch] = []

    # ═══════════════════════════════════════════════════════════
    # 通道間相關性分析
    # ═══════════════════════════════════════════════════════════

    def _update_correlation(self, chunk: np.ndarray):
        """累積資料並定期更新通道間相關係數矩陣。"""
        self._corr_buffer.append(chunk.copy())
        self._corr_buffer_count += chunk.shape[1]

        if self._corr_buffer_count >= self._corr_buffer_size:
            data = np.concatenate(self._corr_buffer, axis=1)  # (n_ch, T)
            n_ch = data.shape[0]

            # 計算 Pearson 相關係數矩陣
            # 先去均值
            data_centered = data - data.mean(axis=1, keepdims=True)
            norms = np.sqrt(np.sum(data_centered ** 2, axis=1, keepdims=True))
            norms = np.maximum(norms, 1e-10)
            data_normed = data_centered / norms

            corr_matrix = data_normed @ data_normed.T  # (n_ch, n_ch)
            self._channel_correlations[:n_ch, :n_ch] = corr_matrix

            self._corr_buffer = []
            self._corr_buffer_count = 0

    def _get_max_correlation(self, ch: int) -> float:
        """取得此通道與其他通道的最大相關係數。-1 表示資料不足。"""
        n_ch = self._channel_correlations.shape[0]
        if np.all(self._channel_correlations == 0):
            return -1.0

        max_corr = 0.0
        for other in range(n_ch):
            if other != ch:
                c = abs(self._channel_correlations[ch, other])
                if c > max_corr:
                    max_corr = c
        return max_corr

    # ═══════════════════════════════════════════════════════════
    # 輔助函式
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """計算峰度 (Fisher's definition, normal=3)。"""
        n = len(x)
        if n < 4:
            return 3.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        m4 = np.mean((x - mean) ** 4)
        return float(m4 / (std ** 4))

    def _score_to_bars(self, score: float) -> tuple:
        for threshold, bars, quality, desc in self._SCORE_THRESHOLDS:
            if score >= threshold:
                return bars, quality, desc
        return 0, 'bad', '極差'
