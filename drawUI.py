import sys
import time
import numpy as np
import pyqtgraph as pg
import queue
import torch

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel
from PySide6.QtCore import QThread, Signal, Slot

from architecture_spikelinear import Model
from read_emg import apply_to_all, notch_harmonics, remove_drift, subsample
from vocoder import Vocoder, StreamingVocoder

from absl import flags
FLAGS = flags.FLAGS

# ---------------------------------------------------------
# 1. 檔案播放執行緒
# ---------------------------------------------------------
class FilePlaybackWorker(QThread):
    update_plot_signal = Signal(np.ndarray)

    def __init__(self, file_path, data_queue):
        super().__init__()
        self.file_path = file_path
        self.data_queue = data_queue
        
        # 【視覺更新頻率】: 控制左側 EMG 波形的流暢度
        # 50ms = 每秒 20 幀 (20 FPS)，視覺上非常平滑
        self.fps_delay = 50 
        
        # 因為訊號已被降採樣至 689.06 Hz，50ms 大約等於 34 個資料點
        self.ui_batch_size = 34
        
        # 【推論打包】: 控制送給模型的資料長度
        # 200ms 在 689Hz 下約等於 138 個點，能產生足夠的幀數瞬間衝破 Vocoder 緩衝限制
        self.inference_batch_size = 138 
        
        self.is_running = True
        self.is_recording = False  
        self.inference_buffer = []
        
        self.data = self._load_and_preprocess_data(file_path)
        self.current_idx = 0
        self.data_length = self.data.shape[1] 

    def _load_and_preprocess_data(self, path):
        if path.endswith('.npy'):
            x = np.load(path)
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

    def run(self):
        self.was_recording = False
        while self.is_running:
            if self.data_length == 0:
                self.msleep(self.fps_delay)
                continue

            # 1. 抓取給介面顯示的一小段資料 (34 個點)
            end_idx = self.current_idx + self.ui_batch_size
            if end_idx <= self.data_length:
                batch_data = self.data[:, self.current_idx:end_idx]
                self.current_idx = end_idx
            else:
                part1 = self.data[:, self.current_idx:]
                part2 = self.data[:, :end_idx - self.data_length]
                batch_data = np.concatenate((part1, part2), axis=1)
                self.current_idx = end_idx - self.data_length

            if self.is_recording:
                emit_data = batch_data
                self.inference_buffer.append(batch_data)
                
                # 2. 計算目前緩衝區累積的資料點數
                current_len = sum(b.shape[1] for b in self.inference_buffer)
                
                # 3. 當累積超過 138 個點 (約 200ms) 時，才打包送給 GPU 推論
                if current_len >= self.inference_batch_size:
                    combined_data = np.concatenate(self.inference_buffer, axis=1)
                    
                    # 防積壓：若 GPU 算不完，丟棄過舊的封包，確保最新狀態
                    if self.data_queue.qsize() > 1:
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            pass
                            
                    self.data_queue.put(combined_data)
                    self.inference_buffer = [] # 清空緩衝區
                    
                self.was_recording = True
            else:
                emit_data = np.zeros_like(batch_data)
                if getattr(self, 'was_recording', False):
                    # 剛按下停止時，清空佇列與緩衝
                    while not self.data_queue.empty():
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            break
                    # 送出足以覆蓋推論批次的全 0 陣列來切斷音訊
                    self.data_queue.put(np.zeros((8, self.inference_batch_size)))
                    self.inference_buffer = []
                    self.was_recording = False

            # 將這 34 個點送給 UI 畫 EMG 波形圖
            self.update_plot_signal.emit(emit_data)
            self.msleep(self.fps_delay)

    @Slot(bool)
    def set_recording(self, state):
        self.is_recording = state

    def stop(self):
        self.is_running = False

# ---------------------------------------------------------
# 2. 模型推論執行緒
# ---------------------------------------------------------
class InferenceWorker(QThread):
    result_signal = Signal(str, float)
    mel_signal = Signal(np.ndarray)
    audio_signal = Signal(np.ndarray)

    def __init__(self, data_queue, weight_path):
        super().__init__()
        self.data_queue = data_queue
        self.weight_path = weight_path
        self.is_running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 【重要修正】這裡不再載入模型，把運算轉移到 run() 中，釋放 UI 執行緒

    def run(self):
        # 1. 系統啟動後，在背景載入模型
        self.result_signal.emit("Status: Loading AI Models...", 0.0)
        try:
            self.model = Model(
                num_features=8, num_speech_features=80, 
                n_phones=48, no_sgc=False, spike_threshold=0.35
            )
            checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Model error: {e}")

        self.streaming_vocoder = None
        if FLAGS.hifigan_checkpoint is not None:
            try:
                base_vocoder = Vocoder(device=str(self.device))
                self.streaming_vocoder = StreamingVocoder(base_vocoder)
                print("Streaming Vocoder loaded successfully!")
            except Exception as e:
                print(f"Vocoder error: {e}")

        self.result_signal.emit("Status: Waiting for input...", 0.0)

        # 2. 進入推論迴圈
        while self.is_running:
            try:
                emg_data = self.data_queue.get(timeout=0.1)
                start_time = time.time()
                
                if np.all(emg_data == 0):
                    mel_features = np.zeros((emg_data.shape[1], 80))
                    self.mel_signal.emit(mel_features)
                    
                    if self.streaming_vocoder is not None:
                        tail_audio = self.streaming_vocoder.flush()
                        if len(tail_audio) > 0:
                            self.audio_signal.emit(tail_audio)
                        self.audio_signal.emit(np.zeros(2000))
                    continue

                input_tensor = torch.from_numpy(emg_data.T).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    session_ids = torch.zeros((1,), dtype=torch.long).to(self.device)
                    pred = self.model(None, input_tensor, session_ids)
                
                latency = time.time() - start_time
                
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
                    mel_chunk = mel_tensor.squeeze(0).to(self.device)
                    audio_chunk = self.streaming_vocoder.synthesize_chunk(mel_chunk)
                    if len(audio_chunk) > 0:
                        self.audio_signal.emit(audio_chunk)

            except queue.Empty:
                self.msleep(10)
                continue
            except Exception as e:
                print(f"推論發生錯誤: {e}")
                self.msleep(10)

    def stop(self):
        self.is_running = False

# ---------------------------------------------------------
# 3. 主 UI 介面
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    recording_signal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("8-Channel EMG & Audio System")
        self.resize(1400, 750)
        self.setStyleSheet("background-color: #2b2b2b;")

        main_layout = QVBoxLayout()
        plot_layout = QHBoxLayout()

        # --- 左側：EMG 波形圖 ---
        self.plot_widget = pg.PlotWidget(title="Real-time 8-Channel EMG")
        self.plot_widget.enableAutoRange(axis='y') 
        # 【效能優化】啟用降採樣與範圍裁剪，降低多線程畫圖負擔
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
        # 【效能優化】預設鎖定顏色對比範圍，避免每幀重複計算
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

        # --- 按鈕與文字區塊 ---
        self.btn_ptt = QPushButton("Click to Start Capture")
        self.btn_ptt.setStyleSheet("background-color: #5cb85c; color: white; font-size: 18px; padding: 15px;")
        self.btn_ptt.clicked.connect(self.toggle_recording)
        main_layout.addWidget(self.btn_ptt)
        
        self.is_recording = False
        self.label_result = QLabel("System Initializing...")
        self.label_result.setStyleSheet("font-size: 24px; color: #ffffff;")
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

        weight_file_path = "model.pt"
        self.inference_thread = InferenceWorker(self.inference_queue, weight_file_path)
        self.inference_thread.result_signal.connect(self.update_result_ui)
        self.inference_thread.mel_signal.connect(self.update_mel_ui)
        self.inference_thread.audio_signal.connect(self.update_audio_ui)
        self.inference_thread.start()

        csv_file_path = "emg_data/silent_parallel_data/5-6_silent/208_emg.npy" 
        self.playback_thread = FilePlaybackWorker(csv_file_path, self.inference_queue)
        self.playback_thread.update_plot_signal.connect(self.update_plot_ui)
        self.recording_signal.connect(self.playback_thread.set_recording)
        self.playback_thread.start()

    def toggle_recording(self, checked=False):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.btn_ptt.setText("Click to Stop Capture")
            self.btn_ptt.setStyleSheet("background-color: #d9534f; color: white; font-size: 18px; padding: 15px;")
            self.label_result.setText("Status: Capturing EMG signals...")
            self.recording_signal.emit(True)
        else:
            self.btn_ptt.setText("Click to Start Capture")
            self.btn_ptt.setStyleSheet("background-color: #5cb85c; color: white; font-size: 18px; padding: 15px;")
            self.label_result.setText("Status: Stopped capturing.")
            self.recording_signal.emit(False)

    @Slot(np.ndarray)
    def update_plot_ui(self, new_data):
        shift = new_data.shape[1]
        self.plot_data[:, :-shift] = self.plot_data[:, shift:]
        self.plot_data[:, -shift:] = new_data
        
        for i in range(8):
            offset = (i - 3.5) * 2
            self.curves[i].setData(self.plot_data[i] + offset)

    @Slot(np.ndarray)
    def update_mel_ui(self, new_mel):
        shift = new_mel.shape[0]
        if shift == 0: return
        self.mel_data[:-shift, :] = self.mel_data[shift:, :]
        self.mel_data[-shift:, :] = new_mel
        # 【效能優化】關閉 autoLevels，不再每幀重新計算矩陣極值
        self.mel_image.setImage(self.mel_data, autoLevels=False)

    @Slot(np.ndarray)
    def update_audio_ui(self, new_audio):
        shift = len(new_audio)
        if shift == 0: 
            return
            
        if shift >= self.audio_buffer_size:
            self.audio_data = new_audio[-self.audio_buffer_size:]
        else:
            self.audio_data[:-shift] = self.audio_data[shift:]
            self.audio_data[-shift:] = new_audio
            
        visual_data = self.audio_data[::80]
        self.audio_curve.setData(visual_data)

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
        event.accept()

if __name__ == "__main__":
    FLAGS(sys.argv) 
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())