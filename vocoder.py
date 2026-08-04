import os
import json
import sys
import numpy as np

import torch

sys.path.append('./hifi_gan')
from env import AttrDict
from models import Generator

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('hifigan_checkpoint', 'hifigan_finetuned/checkpoint', 'filename of hifi-gan generator checkpoint')

HOP_SIZE = 256
TRIM_FRAMES = 8  # 前後截掉的 mel 幀數


class Vocoder(object):
    """原版 Vocoder - 批次推論用"""

    def __init__(self, device='cuda'):
        assert FLAGS.hifigan_checkpoint is not None
        checkpoint_file = FLAGS.hifigan_checkpoint
        config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            hparams = AttrDict(json.load(f))
        self.generator = Generator(hparams).to(device)
        self.generator.load_state_dict(torch.load(checkpoint_file)['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

    def __call__(self, mel_spectrogram):
        '''
            mel_spectrogram should be a tensor of shape (seq_len, 80)
            returns 1d tensor of audio
        '''
        with torch.no_grad():
            mel_spectrogram = mel_spectrogram.T[np.newaxis,:,:]
            audio = self.generator(mel_spectrogram)
        return audio.squeeze()


class StreamingVocoder:
    """
    串流 Vocoder — 處理多幀 + 截掉前後邊緣

    原理 (非常簡單):
    ─────────────────
    1. 每次送 vocoder 時，在左側拼上一段的末尾 mel 作為 context
    2. 跑 vocoder 產生完整音訊
    3. 截掉右邊 TRIM 幀對應的音訊 (右邊界 artifact)
    4. 被截掉的 mel 會成為下一次的左 context，被重新合成 (有了右 context)

    流程圖:
    ────────
    第 1 次: vocoder([mel_1..30])
             output = audio[:-trim]     ← 只保留前面乾淨的
             save mel[-trim:] 給下次

    第 2 次: vocoder([saved_ctx | mel_31..60])
             output = audio[:-trim]     ← ctx 區域音訊被保留 (有右 context 了)
             save mel[-trim:] 給下次

    沒有 crossfade，沒有複雜的 OLA，就是截掉不好的尾巴。
    """

    def __init__(self, vocoder, trim_frames=TRIM_FRAMES):
        self.vocoder = vocoder
        self.trim_frames = trim_frames
        self.trim_samples = trim_frames * HOP_SIZE
        self.mel_left_ctx = None

    def synthesize_chunk(self, mel_chunk):
        """
        Args:
            mel_chunk: tensor (N, 80)
        Returns:
            numpy float32 PCM, or empty array
        """
        device = mel_chunk.device

        # 左側拼接上一次被截掉的 mel context
        if self.mel_left_ctx is not None:
            mel_input = torch.cat([self.mel_left_ctx.to(device), mel_chunk], dim=0)
        else:
            mel_input = mel_chunk

        # 保存右邊 trim_frames 的 mel → 下次的左 context
        save_n = min(self.trim_frames, mel_chunk.shape[0])
        self.mel_left_ctx = mel_chunk[-save_n:].detach().clone()

        # 跑 vocoder
        full_audio = self.vocoder(mel_input).cpu().numpy()

        # 截掉右邊 (有 boundary artifact，下次會帶 context 重新合成)
        if len(full_audio) > self.trim_samples:
            return full_audio[:-self.trim_samples]
        else:
            return full_audio

    def flush(self):
        """最後一段: 重新合成被截掉的尾巴"""
        if self.mel_left_ctx is not None:
            audio = self.vocoder(self.mel_left_ctx).cpu().numpy()
            self.mel_left_ctx = None
            return audio
        return np.array([], dtype=np.float32)
