import string
import os
import subprocess
import tempfile
import struct
import logging

import numpy as np
import librosa
import soundfile as sf
from textgrids import TextGrid
import jiwer
from unidecode import unidecode

import torch
import matplotlib.pyplot as plt

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('normalizers_file', 'normalizers.pkl', 'file with pickled feature normalizers')

phoneme_inventory = ['aa','ae','ah','ao','aw','ax','axr','ay','b','ch','d','dh','dx','eh','el','em','en','er','ey','f','g','hh','hv','ih','iy','jh','k','l','m','n','nx','ng','ow','oy','p','r','s','sh','t','th','uh','uw','v','w','y','z','zh','sil']

def normalize_volume(audio):
    rms = librosa.feature.rms(y=audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms/max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0: # this shouldn't happen too often with the target_rms of 0.2
        audio = audio / max_val
    return audio

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def load_audio(filename, start=None, end=None, max_frames=None, renormalize_volume=False):
    audio, r = sf.read(filename)

    if len(audio.shape) > 1:
        audio = audio[:,0] # select first channel of stero audio
    if start is not None or end is not None:
        audio = audio[start:end]

    if renormalize_volume:
        audio = normalize_volume(audio)
    if r == 16000:
        audio = librosa.resample(audio, orig_sr=16000, target_sr=22050)
    else:
        assert r == 22050
    audio = np.clip(audio, -1, 1) # because resampling sometimes pushes things out of range
    pytorch_mspec = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000, center=False)
    mspec = pytorch_mspec.squeeze(0).T.numpy()
    if max_frames is not None and mspec.shape[0] > max_frames:
        mspec = mspec[:max_frames,:]
    return mspec

# ── FARGAN/LPCNet 特徵提取常數 ────────────────────────────────
LPCNET_SR = 16000           # LPCNet/FARGAN 取樣率
LPCNET_FRAME_SIZE = 160     # 10ms @16kHz
LPCNET_HOP_SIZE = 160       # 幀間隔 (= frame_size, 無重疊)
LPCNET_NB_FEATURES = 20     # 18 Bark cepstral + 2 pitch
LPCNET_NB_BANDS = 18        # Bark-scale cepstral bands
LPCNET_DUMP_NB_FEATURES = 36  # dump_data 輸出的原始特徵數 (取前 20)
_dump_data_disabled = False  # 第一次失敗後設為 True, 避免重複警告

# Bark-scale 中心頻率 (用於 Python fallback)
# 18 個 Bark band 的邊界頻率 (Hz), 近似對應 LPCNet 的 Bark-scale
BARK_BAND_EDGES = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                   1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400,
                   5300, 6400, 7700]


def load_audio_lpcnet(filename, start=None, end=None, max_frames=None,
                      renormalize_volume=False):
    """
    從音訊檔案提取 LPCNet/FARGAN 20-dim 特徵

    輸出: (T, 20) float32
        - dim 0-17: 18 Bark-scale cepstral coefficients
        - dim 18:   pitch period (samples @16kHz, range 32-256)
        - dim 19:   pitch correlation (0-1)

    優先使用 opus dump_data 二進位工具 (精確), 否則用 Python 近似版本。
    特徵會快取為 .npy 檔案以加速後續載入。
    """
    # ── 磁碟快取：避免重複計算 ─────────────────────────────────
    # 快取 key 包含 start/end/renormalize 以區分不同參數的呼叫
    cache_suffix = '_lpcnet_features'
    if start is not None or end is not None:
        cache_suffix += f'_s{start}_e{end}'
    if renormalize_volume:
        cache_suffix += '_renorm'
    cache_path = os.path.splitext(filename)[0] + cache_suffix + '.npy'

    # 檢查快取是否有效 (存在 且 比音訊檔案新)
    if os.path.exists(cache_path):
        try:
            audio_mtime = os.path.getmtime(filename)
            cache_mtime = os.path.getmtime(cache_path)
            if cache_mtime >= audio_mtime:
                features = np.load(cache_path)
                if max_frames is not None and features.shape[0] > max_frames:
                    features = features[:max_frames, :]
                return features
        except Exception:
            pass  # 快取損壞，重新提取

    # ── 正常提取流程 ──────────────────────────────────────────
    audio, r = sf.read(filename)

    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if start is not None or end is not None:
        audio = audio[start:end]
    if renormalize_volume:
        audio = normalize_volume(audio)

    # 統一 resample 到 16kHz
    if r != LPCNET_SR:
        audio = librosa.resample(audio, orig_sr=r, target_sr=LPCNET_SR)
    audio = np.clip(audio, -1, 1).astype(np.float32)

    # 嘗試使用 dump_data 工具 (失敗一次後不再嘗試)
    global _dump_data_disabled
    dump_data_bin = FLAGS.dump_data_bin if hasattr(FLAGS, 'dump_data_bin') else './dump_data'
    if not _dump_data_disabled and os.path.isfile(dump_data_bin) and os.access(dump_data_bin, os.X_OK):
        features = _extract_features_dump_data(audio, dump_data_bin)
    else:
        features = _extract_features_python(audio)

    # ── 存入快取 ──────────────────────────────────────────────
    try:
        np.save(cache_path, features)
    except Exception as e:
        logging.warning(f'無法儲存特徵快取 {cache_path}: {e}')

    if max_frames is not None and features.shape[0] > max_frames:
        features = features[:max_frames, :]

    return features


def _extract_features_dump_data(audio, dump_data_bin):
    """
    使用 opus dump_data 工具提取特徵 (最精確)

    dump_data -test input.s16 features.f32
    輸出: 每幀 36 個 float32, 我們只取前 20 個
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pcm_path = os.path.join(tmpdir, 'input.s16')
        feat_path = os.path.join(tmpdir, 'features.f32')

        # 寫入 16-bit PCM
        pcm_i16 = (audio * 32767).astype(np.int16)
        pcm_i16.tofile(pcm_path)

        # 執行 dump_data
        try:
            subprocess.run(
                [dump_data_bin, '-test', pcm_path, feat_path],
                check=True, capture_output=True, timeout=60
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            global _dump_data_disabled
            _dump_data_disabled = True
            print(f'Warning: dump_data failed ({e}), falling back to Python extraction (this warning will not repeat)')
            return _extract_features_python(audio)

        # 讀取特徵 (每幀 36 個 float32)
        features_raw = np.fromfile(feat_path, dtype=np.float32)
        nb_frames = len(features_raw) // LPCNET_DUMP_NB_FEATURES
        features_raw = features_raw[:nb_frames * LPCNET_DUMP_NB_FEATURES]
        features_raw = features_raw.reshape(nb_frames, LPCNET_DUMP_NB_FEATURES)

        # 只取前 20 維
        features = features_raw[:, :LPCNET_NB_FEATURES]

    return features.astype(np.float32)


def _extract_features_python(audio):
    """
    純 Python 近似 LPCNet 特徵提取 (fallback)

    使用 librosa 計算:
      - 18-dim Bark-scale cepstral coefficients (近似)
      - pitch period + pitch correlation

    注意: 這不是 LPCNet 的精確實作, 但在沒有 dump_data 時可以用來
    做初步實驗。建議正式訓練時使用 dump_data。
    """
    sr = LPCNET_SR
    frame_length = LPCNET_FRAME_SIZE    # 160 samples = 10ms
    hop_length = LPCNET_FRAME_SIZE      # 無重疊

    # ── Bark-scale cepstral coefficients ────────────────────────
    # 使用較接近 Bark 尺度的 mel filterbank (fmin=0, fmax=8000, n_mels=18)
    # Bark 和 mel 在低頻區很接近，在高頻區有差異但可接受
    n_fft = 320  # 20ms window for better frequency resolution
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft,
        hop_length=hop_length, win_length=frame_length,
        n_mels=LPCNET_NB_BANDS,
        fmin=0, fmax=sr // 2,
        center=False
    )

    # Log compress + DCT → cepstral coefficients
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    # DCT (type II) 轉為 cepstral domain, 保留 18 個係數
    from scipy.fftpack import dct
    bark_cepstral = dct(log_mel.T, type=2, norm='ortho', axis=1)[:, :LPCNET_NB_BANDS]
    # shape: (T, 18)

    # ── Pitch 提取 ─────────────────────────────────────────────
    # 使用 librosa pyin 做 F0 估計
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=100, fmax=500, sr=sr,
        frame_length=frame_length * 2,
        hop_length=hop_length,
        center=False
    )

    # 確保長度對齊
    n_frames = bark_cepstral.shape[0]
    if len(f0) > n_frames:
        f0 = f0[:n_frames]
        voiced_prob = voiced_prob[:n_frames]
    elif len(f0) < n_frames:
        pad_len = n_frames - len(f0)
        f0 = np.pad(f0, (0, pad_len), mode='edge')
        voiced_prob = np.pad(voiced_prob, (0, pad_len), mode='edge')

    # F0 → pitch period (samples @16kHz)
    # LPCNet pitch period 範圍: 32-256 samples
    pitch_period = np.where(
        np.isnan(f0) | (f0 < 1),
        256.0,  # unvoiced → max period
        np.clip(sr / f0, 32, 256)
    )

    # pitch correlation (voiced probability)
    pitch_corr = np.where(np.isnan(voiced_prob), 0.0, voiced_prob)
    pitch_corr = np.clip(pitch_corr, 0, 1)

    # ── 組合 20-dim 特徵 ───────────────────────────────────────
    features = np.zeros((n_frames, LPCNET_NB_FEATURES), dtype=np.float32)
    features[:, :LPCNET_NB_BANDS] = bark_cepstral
    features[:, LPCNET_NB_BANDS] = pitch_period
    features[:, LPCNET_NB_BANDS + 1] = pitch_corr

    return features


def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w

def get_emg_features(emg_data, debug=False):
    xs = emg_data - emg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:,i]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(w, frame_length=16, hop_length=6).mean(axis=0)
        p_w = librosa.feature.rms(y=w, frame_length=16, hop_length=6, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(y=r, frame_length=16, hop_length=6, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=16, hop_length=6, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=16, hop_length=6).mean(axis=0)

        s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))
        # s has feature dimension first and time second

        if debug:
            plt.subplot(7,1,1)
            plt.plot(x)
            plt.subplot(7,1,2)
            plt.plot(w_h)
            plt.subplot(7,1,3)
            plt.plot(p_w)
            plt.subplot(7,1,4)
            plt.plot(p_r)
            plt.subplot(7,1,5)
            plt.plot(z_p)
            plt.subplot(7,1,6)
            plt.plot(r_h)

            plt.subplot(7,1,7)
            plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')

            plt.show()

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)

class FeatureNormalizer(object):
    def __init__(self, feature_samples, share_scale=False):
        """ features_samples should be list of 2d matrices with dimension (time, feature) """
        feature_samples = np.concatenate(feature_samples, axis=0)
        self.feature_means = feature_samples.mean(axis=0, keepdims=True)
        if share_scale:
            self.feature_stddevs = feature_samples.std()
        else:
            self.feature_stddevs = feature_samples.std(axis=0, keepdims=True)

    def normalize(self, sample):
        sample -= self.feature_means
        sample /= self.feature_stddevs
        return sample

    def inverse(self, sample):
        sample = sample * self.feature_stddevs
        sample = sample + self.feature_means
        return sample

def combine_fixed_length(tensor_list, length):
    total_length = sum(t.size(0) for t in tensor_list)
    if total_length % length != 0:
        pad_length = length - (total_length % length)
        tensor_list = list(tensor_list) # copy
        tensor_list.append(torch.zeros(pad_length,*tensor_list[0].size()[1:], dtype=tensor_list[0].dtype, device=tensor_list[0].device))
        total_length += pad_length
    tensor = torch.cat(tensor_list, 0)
    n = total_length // length
    return tensor.view(n, length, *tensor.size()[1:])

def decollate_tensor(tensor, lengths):
    b, s, d = tensor.size()
    tensor = tensor.view(b*s, d)
    results = []
    idx = 0
    for length in lengths:
        assert idx + length <= b * s
        results.append(tensor[idx:idx+length])
        idx += length
    return results

def splice_audio(chunks, overlap):
    chunks = [c.copy() for c in chunks] # copy so we can modify in place

    assert np.all([c.shape[0]>=overlap for c in chunks])

    result_len = sum(c.shape[0] for c in chunks) - overlap*(len(chunks)-1)
    result = np.zeros(result_len, dtype=chunks[0].dtype)

    ramp_up = np.linspace(0,1,overlap)
    ramp_down = np.linspace(1,0,overlap)

    i = 0
    for chunk in chunks:
        l = chunk.shape[0]

        # note: this will also fade the beginning and end of the result
        chunk[:overlap] *= ramp_up
        chunk[-overlap:] *= ramp_down

        result[i:i+l] += chunk
        i += l-overlap

    return result

def print_confusion(confusion_mat, n=10):
    # axes are (pred, target)
    target_counts = confusion_mat.sum(0) + 1e-4
    aslist = []
    for p1 in range(len(phoneme_inventory)):
        for p2 in range(p1):
            if p1 != p2:
                aslist.append(((confusion_mat[p1,p2]+confusion_mat[p2,p1])/(target_counts[p1]+target_counts[p2]), p1, p2))
    aslist.sort()
    aslist = aslist[-n:]
    max_val = aslist[-1][0]
    min_val = aslist[0][0]
    val_range = max_val - min_val
    print('Common confusions (confusion, accuracy)')
    for v, p1, p2 in aslist:
        p1s = phoneme_inventory[p1]
        p2s = phoneme_inventory[p2]
        print(f'{p1s} {p2s} {v*100:.1f} {(confusion_mat[p1,p1]+confusion_mat[p2,p2])/(target_counts[p1]+target_counts[p2])*100:.1f}')

def read_phonemes(textgrid_fname, max_len=None, frame_rate=None):
    if frame_rate is None:
        frame_rate = 86.133
    tg = TextGrid(textgrid_fname)
    phone_ids = np.zeros(int(tg['phones'][-1].xmax * frame_rate) + 1, dtype=np.int64)
    phone_ids[:] = -1
    phone_ids[-1] = phoneme_inventory.index('sil') # make sure list is long enough to cover full length of original sequence
    for interval in tg['phones']:
        phone = interval.text.lower()
        if phone in ['', 'sp', 'spn']:
            phone = 'sil'
        if phone[-1] in string.digits:
            phone = phone[:-1]
        ph_id = phoneme_inventory.index(phone)
        phone_ids[int(interval.xmin * frame_rate):int(interval.xmax * frame_rate)] = ph_id
    assert (phone_ids >= 0).all(), 'missing aligned phones'

    if max_len is not None:
        phone_ids = phone_ids[:max_len]
        assert phone_ids.shape[0] == max_len
    return phone_ids

class TextTransform(object):
    def __init__(self):
        self.transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
        self.chars = string.ascii_lowercase+string.digits+' '

    def clean_text(self, text):
        text = unidecode(text)
        text = self.transformation(text)
        return text

    def text_to_int(self, text):
        text = self.clean_text(text)
        return [self.chars.index(c) for c in text]

    def int_to_text(self, ints):
        return ''.join(self.chars[i] for i in ints)
