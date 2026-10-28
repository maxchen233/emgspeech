import os
import json
import sys
import logging
import numpy as np

import torch

sys.path.append('./hifi_gan')
from env import AttrDict
from models import Generator

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('hifigan_checkpoint','hifigan_finetuned/checkpoint', 'filename of hifi-gan generator checkpoint')
flags.DEFINE_string('vocoder_onnx', None,
                    '剪枝+量化 ONNX vocoder 路徑 (e.g. hifigan_pruned_qat_qdq.onnx). '
                    '設定後將使用 ONNXVocoder 取代原版 PyTorch HiFi-GAN')

HOP_SIZE = 256
TRIM_FRAMES = 8  # 前後截掉的 mel 幀數


class Vocoder(object):
    """原版 Vocoder - 批次推論用 (同時支援原版 checkpoint 與 QAT checkpoint)"""

    def __init__(self, device='cuda'):
        assert FLAGS.hifigan_checkpoint is not None
        checkpoint_file = FLAGS.hifigan_checkpoint

        # ── 尋找 config.json (支援 QAT checkpoint 不在 hifigan_finetuned/ 的情況) ──
        config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
        if not os.path.exists(config_file):
            # 回退到標準路徑
            for fallback in ['hifigan_finetuned/config.json', 'hifi_gan/config_v1.json']:
                if os.path.exists(fallback):
                    config_file = fallback
                    break

        with open(config_file) as f:
            hparams = AttrDict(json.load(f))

        self.generator = Generator(hparams).to(device)
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        gen_state = ckpt['generator'] if 'generator' in ckpt else ckpt

        # ── 自動偵測 checkpoint 格式 ──
        # 原版 checkpoint: 含 weight_norm 參數化 (key 含 weight_g / weight_v)
        # QAT checkpoint:  已移除 weight_norm，只有乾淨的 weight key
        has_weight_norm_keys = any(
            k.endswith('.weight_g') or k.endswith('.weight_v') for k in gen_state.keys()
        )

        if has_weight_norm_keys:
            # 原版格式：先載入 (含 weight_norm)，再移除 weight_norm
            self.generator.load_state_dict(gen_state)
            self.generator.eval()
            self.generator.remove_weight_norm()
        else:
            # QAT 格式：先移除 weight_norm (讓 key 變成 plain weight)，再載入
            self.generator.remove_weight_norm()
            self.generator.load_state_dict(gen_state)
            self.generator.eval()
            import logging
            logging.info("  ✅ 已載入 QAT HiFi-GAN vocoder (無 weight_norm)")

    def __call__(self, mel_spectrogram):
        '''
            mel_spectrogram should be a tensor of shape (seq_len, 80)
            returns 1d tensor of audio  
        '''
        with torch.no_grad():
            mel_spectrogram = mel_spectrogram.T[np.newaxis,:,:]
            audio = self.generator(mel_spectrogram)
        return audio.squeeze()


class ONNXVocoder(object):
    """
    ONNX Vocoder — 載入剪枝+量化後的 ONNX HiFi-GAN 模型 (QDQ 格式)

    與原版 Vocoder 完全相同的 __call__ 介面:
        輸入: mel_spectrogram tensor (seq_len, 80)
        輸出: 1d tensor of audio

    效能最佳化:
        - 多執行緒卷積平行化 (自動偵測 CPU 核心數)
        - Session warmup 消除首次推論延遲
        - CUDA EP 時使用 IO Binding 避免 CPU↔GPU 複製
        - 記憶體 arena 與 pattern 優化緩衝區重用
    """

    def __init__(self, onnx_path=None, device='cuda'):
        if not HAS_ONNXRUNTIME:
            raise ImportError(
                "ONNXVocoder 需要 onnxruntime。請安裝: pip install onnxruntime "
                "或 pip install onnxruntime-gpu (CUDA 版)"
            )

        onnx_path = onnx_path or FLAGS.vocoder_onnx
        if onnx_path is None or not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"找不到 ONNX vocoder 模型: {onnx_path}\n"
                f"請用 --vocoder_onnx 指定路徑"
            )

        # ── 選擇最佳 ExecutionProvider ───────────────────────
        available_providers = ort.get_available_providers()
        if 'TensorrtExecutionProvider' in available_providers and 'cuda' in device:
            providers = [
                ('TensorrtExecutionProvider', {
                    'trt_max_workspace_size': 1 << 30,  # 1 GB
                    'trt_fp16_enable': True,
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        elif 'CUDAExecutionProvider' in available_providers and 'cuda' in device:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # ── Session 最佳化選項 ────────────────────────────────
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True       # 緩衝區配置模式重用
        sess_options.enable_cpu_mem_arena = True      # CPU 記憶體池
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 多執行緒: 讓卷積層可以平行計算 (比單執行緒快 3~5 倍)
        import multiprocessing
        n_cores = multiprocessing.cpu_count() or 1
        n_threads = max(1, min(n_cores, 4))
        sess_options.intra_op_num_threads = n_threads
        sess_options.inter_op_num_threads = 1  # 串流場景 op 間順序執行即可

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name   # 'mel'
        self.output_name = self.session.get_outputs()[0].name  # 'audio'
        self.device = device

        actual_provider = self.session.get_providers()[0]
        self.use_cuda = ('CUDA' in actual_provider or 'Tensorrt' in actual_provider)

        # ── IO Binding (CUDA EP 專用，避免 CPU↔GPU 複製) ─────
        self._io_binding = None
        if self.use_cuda:
            try:
                self._io_binding = self.session.io_binding()
            except Exception:
                self._io_binding = None

        # ── Warmup: 消除首次推論 JIT 與記憶體分配延遲 ─────────
        import time
        dummy = np.zeros((1, 80, 32), dtype=np.float32)
        t0 = time.perf_counter()
        self.session.run(None, {self.input_name: dummy})
        warmup_ms = (time.perf_counter() - t0) * 1000

        model_size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        logging.info(
            f"  ✅ ONNX Vocoder 載入完成: {onnx_path} "
            f"({model_size_mb:.2f} MB)"
        )
        logging.info(
            f"     provider={actual_provider} | "
            f"threads={n_threads} | "
            f"io_binding={'ON' if self._io_binding else 'OFF'} | "
            f"warmup={warmup_ms:.1f}ms"
        )

    def __call__(self, mel_spectrogram):
        '''
            mel_spectrogram should be a tensor of shape (seq_len, 80)
            returns 1d tensor of audio
        '''
        # 轉換為 ONNX 預期格式: (1, 80, seq_len)
        if isinstance(mel_spectrogram, torch.Tensor):
            mel_np = mel_spectrogram.detach().cpu().numpy()
        else:
            mel_np = mel_spectrogram

        audio_np = self._run_onnx(mel_np.T[np.newaxis, :, :].astype(np.float32))
        return torch.from_numpy(audio_np).float()

    def _run_onnx(self, mel_input):
        """
        核心推論: mel_input shape (1, 80, T) → audio shape (samples,)
        CUDA EP 使用 IO Binding 避免多餘的 host↔device 複製。
        """
        if self._io_binding is not None:
            try:
                binding = self._io_binding
                mel_ort = ort.OrtValue.ortvalue_from_numpy(mel_input, 'cuda', 0)
                binding.bind_ortvalue_input(self.input_name, mel_ort)
                binding.bind_output(self.output_name, 'cuda')
                self.session.run_with_iobinding(binding)
                audio = binding.get_outputs()[0].numpy().squeeze()
                binding.clear_binding_inputs()
                binding.clear_binding_outputs()
                return audio
            except Exception:
                # fallback 到一般推論
                pass

        outputs = self.session.run(None, {self.input_name: mel_input})
        return outputs[0].squeeze()


class StreamingONNXVocoder:
    """
    ONNX 專用串流 Vocoder — 完全在 numpy 空間操作

    與 StreamingVocoder 提供相同介面 (synthesize_chunk / flush)，
    但內部完全避免 torch.Tensor 轉換，消除每次推論的:
        torch.Tensor → numpy → ONNX → numpy → torch.Tensor → .cpu() → .numpy()
    熱路徑只有:
        numpy concatenate → ONNX → numpy slice

    典型加速: 相比 StreamingVocoder + ONNXVocoder 組合快 30~50%
    """

    def __init__(self, onnx_vocoder, trim_frames=TRIM_FRAMES):
        assert isinstance(onnx_vocoder, ONNXVocoder)
        self.vocoder = onnx_vocoder
        self.trim_frames = trim_frames
        self.trim_samples = trim_frames * HOP_SIZE
        self._mel_left_ctx = None   # numpy (N, 80) float32

    def synthesize_chunk(self, mel_chunk):
        """
        Args:
            mel_chunk: tensor (N, 80) 或 numpy (N, 80)
        Returns:
            numpy float32 PCM, or empty array
        """
        # ── 轉 numpy (如果是 torch tensor) ────────────────
        if isinstance(mel_chunk, torch.Tensor):
            mel_np = mel_chunk.detach().cpu().numpy().astype(np.float32)
        else:
            mel_np = np.ascontiguousarray(mel_chunk, dtype=np.float32)

        # ── 左側拼接上一段被截掉的 mel context ─────────────
        if self._mel_left_ctx is not None:
            mel_input = np.concatenate([self._mel_left_ctx, mel_np], axis=0)
        else:
            mel_input = mel_np

        # ── 保存右邊 trim_frames 給下次 context ───────────
        save_n = min(self.trim_frames, mel_np.shape[0])
        self._mel_left_ctx = mel_np[-save_n:].copy()

        # ── ONNX 推論 (直接 numpy → numpy) ────────────────
        mel_transposed = np.ascontiguousarray(
            mel_input.T[np.newaxis, :, :], dtype=np.float32
        )
        full_audio = self.vocoder._run_onnx(mel_transposed)

        # ── 截掉右邊 boundary artifact ────────────────────
        if len(full_audio) > self.trim_samples:
            return full_audio[:-self.trim_samples].astype(np.float32)
        return full_audio.astype(np.float32)

    def flush(self):
        """最後一段: 重新合成被截掉的尾巴"""
        if self._mel_left_ctx is not None:
            mel_transposed = np.ascontiguousarray(
                self._mel_left_ctx.T[np.newaxis, :, :], dtype=np.float32
            )
            audio = self.vocoder._run_onnx(mel_transposed).astype(np.float32)
            self._mel_left_ctx = None
            return audio
        return np.array([], dtype=np.float32)


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

