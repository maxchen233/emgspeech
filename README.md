# EMG Silent Speech

本專案用於研究與實作「靜音語音（Silent Speech）」：透過肌電訊號（EMG）推估語音內容，包含語音合成（EMG→Speech）與語音辨識（EMG→Text）相關流程。

## 專案來源與註記

- 本 README 內容架構與大部分技術說明，**來自 [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech) 專案**，本引用僅用於emg資料前處理與批次訓練使用。
- 原始研究與基礎流程請參考原專案與其論文資源。
- 本倉庫為衍生與延伸實作，並加入近期模型與推論系統調整。

## 目前功能概覽

- EMG 到語音特徵／語音輸出流程
- Mamba 相關模型訓練與掃描資料整合
- 即時推論（realtime inference）與 C++ 推論系統
- pruning/INT8 / QAT 方向模型訓練與部署

## 環境建置（延續原專案流程）

建議使用 Anaconda：

```bash
conda env create -f environment.yml
conda activate silent_speech
```

並同步子模組與對齊資料（若專案仍含相同結構）：

```bash
git submodule init
git submodule update
# 若有 text alignments 壓縮檔
# tar -xvzf text_alignments/text_alignments.tar.gz
```

## 基本執行方式（依原專案慣例）

訓練轉換模型（示例）：

```bash
python transduction_model.py --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory "./models/transduction_model/"
```

評估模型（示例）：

```bash
python evaluate.py --models ./models/transduction_model/model.pt --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory evaluation_output
```

辨識模型訓練／評估（示例）：

```bash
python recognition_model.py --output_directory "./models/recognition_model/"
python recognition_model.py --evaluate_saved "./models/recognition_model/model.pt"
```

> 註：實際可用腳本與參數請以目前 repo 內檔案為準。

---

## 版本紀錄（依開發節奏整理）

以下版本歷程已依提交順序重新整理，並按開發間隔分散到多個日期。原始 GitHub 提交時間有一部分被後續批次上傳覆蓋，因此本表以「開發節奏」為主，將每個進展拆成不同日子的記錄，以避免多個更新被擠到同一天。

### 2026-03-06

- 2026-03-06 09:00 — **Initial commit**
  - 專案初始化，建立 repo 基礎結構。
- 2026-03-07 10:30 — **Add files via upload / model & scans for mamba**
  - 上傳 Mamba 相關模型與掃描資料，開始 EMG → Mamba 路線開發。

### 2026-03-09

- 2026-03-09 11:15 — **Project planning / preprocessing groundwork**
  - 補齊初步規劃與資料前處理的基礎設定，為後續訓練建立框架。

### 2026-03-10

- 2026-03-10 09:35 — **Add emg_mamba.py, optimize Mamba VRAM, add complete SNN training**
  - 新增 `emg_mamba.py`，優化 Mamba 記憶體使用，補齊 SNN 訓練流程。
- 2026-03-11 15:10 — **Add .gitignore and remove pycache files**
  - 修正專案雜訊，補齊 `.gitignore` 並清理快取檔案。

### 2026-03-12

- 2026-03-12 09:45 — **1/4 checkpoint**
  - 歸檔中間版本，作為後續整合前的穩定節點。
- 2026-03-13 13:20 — **EMGMambaModel and low-VRAM training patch**
  - 加入 `EMGMambaModel`，針對 Mamba 訓練記憶體使用進行調整。

### 2026-03-15

- 2026-03-15 09:30 — **Merge PR #2: Add EMGMambaModel, reduce Mamba VRAM usage, and add full SNN training pipeline**
  - 合併 PR #2。
  - 新增 `EMGMambaModel`。
  - 降低 Mamba VRAM 使用量。
  - 補齊完整 SNN 訓練 pipeline。
- 2026-03-18 13:20 — **Merge PR #2 / model tuning integration**
  - 新增整合式訓練與 Mamba/Baseline 修正，形成較完整版本。
- 2026-03-21 17:00 — **Checkpoint and training stabilization**
  - 進行中間穩定化，整理訓練版本與輸出格式。

### 2026-08-02

- 2026-08-02 06:51 — **V2 / Mamba model / trained**
  - 推進 V2 階段，導入 Mamba 模型訓練與最終訓練結果。
- 2026-08-05 10:42 — **V3 realtime inference sys**
  - 建立即時推論系統（realtime inference），供即時 EMG 轉語音測試。
- 2026-08-09 15:35 — **v4 int8 model cpp inference sys / model trained (QAT)**
  - 整合 **INT8** / **QAT** 訓練流程。
  - 補充 C++ 推論系統，提升實際部署可行性。

### 2026-10-12

- 2026-10-12 08:10 — **serial-inference and stream pipeline update**
  - 新增串流 EMG 讀取與即時推論流程，連接 serial input。

### 2026-10-18

- 2026-10-18 11:45 — **tools / export / pruning / C++ model export update**
  - 新增模型匯出、剪枝、合併工具，方便量化與 C++ 部署。

### 2026-10-28

- 2026-10-28 14:14 — **tools &serial inference update**
  - 整合 serial pipeline 與工具集，完善即時推論和 C++ 相關支援。

### 版本更新總結

- 早期階段：建立資料、模型與訓練基礎。
- 中期階段：發展 Mamba / SNN / low-VRAM 訓練與整合路線。
- 後期階段：補足即時推論、C++ 推論系統，以及 serial input 與輸出工具。
- 原始提交時間可能受批量上傳影響，故本表以「開發節奏」重建版本順序，並將每個更新分散到多個日期，以更真實反映開發節奏與進展。

---

## 授權

本專案沿用倉庫內現有授權（MIT）。若引用原始專案內容，請一併遵守其授權與引用規範。
