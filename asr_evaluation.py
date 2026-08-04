import os
import logging

import whisper
import jiwer
import soundfile as sf
import numpy as np
from unidecode import unidecode
import librosa
import tqdm
import torch

def evaluate(testset, audio_directory):
    # 自動偵測要使用 GPU 還是 CPU 來跑評估
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 載入 Whisper 模型
    # 這裡預設使用 'base' 模型，兼顧速度與準確率。
    # 如果你的 GPU 夠力，也可以改用 'small' 或 'medium' 來獲得更精準的 WER 評估
    logging.info("Loading Whisper 'base' model for evaluation...")
    model = whisper.load_model("base").to(device)
    
    predictions = []
    targets = []
    
    for i, datapoint in enumerate(tqdm.tqdm(testset, 'Evaluate outputs', disable=None)):
        # 讀取合成出來的音檔
        audio_path = os.path.join(audio_directory, f'example_output_{i}.wav')
        audio, rate = sf.read(audio_path)
        
        # Whisper 強制要求 16000Hz 的採樣率
        if rate != 16000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            
        # 關鍵差異：DeepSpeech 需要轉成 int16，但 Whisper 只需要標準的 float32 格式
        audio = audio.astype(np.float32)
        
        # 執行 Whisper 語音辨識
        # fp16=True 可以在 GPU 上加速推論並節省顯存
        result = model.transcribe(audio, fp16=torch.cuda.is_available())
        text = result["text"].strip()
        predictions.append(text)
        
        target_text = unidecode(datapoint['text'])
        targets.append(target_text)
        
    # 文字正規化：去除標點符號、轉成全小寫，讓評估更公平
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    
    # 寫入 Log 紀錄
    logging.info(f'targets: {targets}')
    logging.info(f'predictions: {predictions}')
    logging.info(f'wer: {jiwer.wer(targets, predictions)}')