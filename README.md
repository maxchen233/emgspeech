# EMG Silent Speech（中文說明）

本專案用於研究與實作「靜音語音（Silent Speech）」：透過肌電訊號（EMG）推估語音內容，包含語音合成（EMG→Speech）與語音辨識（EMG→Text）相關流程。

## 專案來源與註記

- 本 README 內容架構與大部分技術說明，**來自 [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech) 專案**，本引用僅用於emg資料前處理與批次訓練使用。
- 原始研究與基礎流程請參考原專案與其論文資源。
- 本倉庫為衍生與延伸實作，並加入近期模型與推論系統調整。

## 目前功能概覽

- EMG 到語音特徵／語音輸出流程
- EMG 到文字辨識流程（CTC 路徑）
- Mamba 相關模型訓練與掃描資料整合
- 即時推論（realtime inference）與 C++ 推論系統嘗試
- INT8 / QAT 方向模型訓練與部署嘗試

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

## 版本紀錄（依近期 commit 整理）

以下依時間由新到舊整理：

### 2026-08-04

- **v4 int8 model cpp inference sys / model trained (QAT)**  
  Commit: `6e35ebb8cdde3cc09e598e6549a368b7471b92b9`  
  重點：
  - 新增／調整 V4 路線，聚焦 **INT8** 與 **QAT（Quantization-Aware Training）**。
  - 針對 **C++ inference system** 方向進行整合與訓練結果提交。

- **V3 realtime inference sys**  
  Commit: `24c73d4c79d5e4e9b5b8df6b0d58f526b006f014`  
  重點：
  - 建立或更新 **V3 即時推論系統（realtime inference）**。

- **V2 / mamba model / trained**  
  Commit: `ef35b5fc6c4752738648d1e6a3193a09012ff293`  
  重點：
  - 推進 **V2** 階段，導入與訓練 **Mamba model**。

### 2026-03-15

- **1/4**  
  Commit: `05763465255fada0bd1c9d9fb12a73261e030b29`  
  重點：
  - 階段性提交（可視為後續優化前的中繼節點）。

- **Merge PR #2: Add EMGMambaModel, reduce Mamba VRAM usage, and add full SNN training pipeline**  
  Commit: `60bcd6d25a30e79686fd754755bd7ff0ce4a268e`  
  重點：
  - 合併 PR #2。
  - 新增 **EMGMambaModel**。
  - 降低 **Mamba VRAM** 使用量。
  - 加入完整 **SNN training pipeline**。

- **Add .gitignore and remove pycache files**  
  Commit: `32358555e0088860eef80dc826291f06e6f9aa67`  
  重點：
  - 補齊 `.gitignore`。
  - 清理 `__pycache__` 類型檔案。

- **Add emg_mamba.py, optimize Mamba VRAM, add complete SNN training**  
  Commit: `8b75bd4d045dd8540a1ce62af59e082756528c8d`  
  重點：
  - 新增 `emg_mamba.py`。
  - 優化 Mamba 記憶體使用。

- **Initial plan**  
  Commit: `b1a166211c541663c905ac07aceaf1d5c23579d0`  
  重點：
  - 初始規劃提交。

### 2026-03-06

- **Add files via upload / model & scans for mamba**  
  Commit: `3a0a1eb0f1da865f493b685f18c825890e163f7b`  
  重點：
  - 上傳 Mamba 相關模型與掃描資料。

- **Initial commit**  
  Commit: `6fa2df4f624de47789323453beac9a64da2e9896`  
  重點：
  - 專案初始化。

---

## 授權

本專案沿用倉庫內現有授權（MIT）。若引用原始專案內容，請一併遵守其授權與引用規範。
