import re
import os
import numpy as np
import random
import scipy
import json
import copy
import sys
import pickle
import string
import logging
from functools import lru_cache
from copy import copy

import torch

from data_utils import load_audio, get_emg_features, FeatureNormalizer, phoneme_inventory, read_phonemes, TextTransform

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('remove_channels', [], 'channels to remove')
flags.DEFINE_list('silent_data_directories', ['./emg_data/silent_parallel_data'], 'silent data locations')
flags.DEFINE_list('voiced_data_directories', ['./emg_data/voiced_parallel_data','./emg_data/nonparallel_data'], 'voiced data locations')
flags.DEFINE_string('testset_file', 'testset_largedev.json', 'file with testset indices')
flags.DEFINE_string('text_align_directory', 'text_alignments', 'directory with alignment files')
flags.DEFINE_boolean('make_normalizers', False, 'compute and save normalizers')

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)

def load_utterance(base_dir, index, limit_length=False, debug=False, text_align_directory=None):
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    before = os.path.join(base_dir, f'{index-1}_emg.npy')
    after = os.path.join(base_dir, f'{index+1}_emg.npy')
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]
    # 80-bin mel: 86.13 fps (hop=256 @22.05 kHz).
    emg_orig = apply_to_all(subsample, x, 689.06, 1000)
    x = apply_to_all(subsample, x, 516.79, 1000)
    emg = x

    for c in FLAGS.remove_channels:
        emg[:,int(c)] = 0
        emg_orig[:,int(c)] = 0

    emg_features = get_emg_features(emg)

    audio_path = os.path.join(base_dir, f'{index}_audio_clean.flac')
    mfccs = load_audio(
        audio_path,
        max_frames=min(emg_features.shape[0], 800 if limit_length else float('inf'))
    )

    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[:mfccs.shape[0],:]
    assert emg_features.shape[0] == mfccs.shape[0]
    emg = emg[6:6+6*emg_features.shape[0],:]
    emg_orig = emg_orig[8:8+8*emg_features.shape[0],:]
    assert emg.shape[0] == emg_features.shape[0]*6

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = f'{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid'
    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname, mfccs.shape[0])
    else:
        phonemes = np.zeros(mfccs.shape[0], dtype=np.int64)+phoneme_inventory.index('sil')

    return mfccs, emg_features, info['text'], (info['book'],info['sentence_index']), phonemes, emg_orig.astype(np.float32)

class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory

class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            directory_info, file_idx = self.dataset.example_indices[idx]
            with open(os.path.join(directory_info.directory, f'{file_idx}_info.json')) as f:
                info = json.load(f)
            if not np.any([l in string.ascii_letters for l in info['text']]):
                continue
            length = sum([emg_len for emg_len, _, _ in info['chunks']])
            if length > self.max_len:
                logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch

class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, emg_file=None, limit_length=False, dev=False, test=False, no_testset=False, no_normalizers=False):

        self.text_align_directory = getattr(FLAGS, 'text_align_directory', 'text_alignments')

        # ── 檢查是否已指定單一/特定 EMG 檔案 ──────────────────────────
        # 可由參數 emg_file、base_dir 為檔案路徑、或由 CLI --emg_file 提供
        if emg_file is None:
            emg_file = getattr(FLAGS, 'emg_file', None)
        if emg_file is None and base_dir is not None and (os.path.isfile(base_dir) or str(base_dir).endswith('.npy')):
            emg_file = base_dir
        if emg_file is not None and os.path.isdir(emg_file):
            npy_files = [os.path.join(emg_file, f) for f in sorted(os.listdir(emg_file)) if f.endswith('.npy')]
            if npy_files:
                emg_file = npy_files[0]

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            norm_file = getattr(FLAGS, 'normalizers_file', 'normalizers.pkl')
            if os.path.exists(norm_file):
                with open(norm_file, 'rb') as f:
                    self.mfcc_norm, self.emg_norm = pickle.load(f)
            else:
                self.mfcc_norm, self.emg_norm = None, None
        else:
            self.mfcc_norm, self.emg_norm = None, None

        self.limit_length = limit_length
        self.text_transform = TextTransform()
        self.num_speech_features = 80
        self.num_features = 8
        self.num_sessions = 1

        # 若已指定檔案：直接進入單一檔案模式，完全不掃描整個資料夾與 testset
        if emg_file is not None:
            self.emg_file = emg_file
            self.example_indices = [(EMGDirectory(0, os.path.dirname(emg_file) or '.', False), 0)]
            self.voiced_data_locations = {}
            return

        # ── 一般完整資料集模式 ───────────────────────────────────────
        if no_testset:
            devset = []
            testset = []
        else:
            testset_file = getattr(FLAGS, 'testset_file', 'testset_largedev.json')
            if os.path.exists(testset_file):
                with open(testset_file) as f:
                    testset_json = json.load(f)
                    devset = testset_json.get('dev', [])
                    testset = testset_json.get('test', [])
            else:
                devset = []
                testset = []

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            for sd in getattr(FLAGS, 'silent_data_directories', []):
                if os.path.exists(sd) and os.path.isdir(sd):
                    for session_dir in sorted(os.listdir(sd)):
                        s_path = os.path.join(sd, session_dir)
                        if os.path.isdir(s_path):
                            directories.append(EMGDirectory(len(directories), s_path, True))

            has_silent = any(d.silent for d in directories)
            for vd in getattr(FLAGS, 'voiced_data_directories', []):
                if os.path.exists(vd) and os.path.isdir(vd):
                    for session_dir in sorted(os.listdir(vd)):
                        s_path = os.path.join(vd, session_dir)
                        if os.path.isdir(s_path):
                            directories.append(EMGDirectory(len(directories), s_path, False, exclude_from_testset=has_silent))

        self.example_indices = []
        self.voiced_data_locations = {} # map from book/sentence_index to directory_info/index
        for directory_info in directories:
            if not os.path.exists(directory_info.directory):
                continue
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.json', fname)
                if m is not None:
                    idx_str = m.group(1)
                    info_path = os.path.join(directory_info.directory, fname)
                    with open(info_path) as f:
                        info = json.load(f)
                        if info['sentence_index'] >= 0: # boundary clips of silence are marked -1
                            location_in_testset = [info['book'], info['sentence_index']] in testset
                            location_in_devset = [info['book'], info['sentence_index']] in devset
                            if (test and location_in_testset and not directory_info.exclude_from_testset) \
                                    or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                                    or (not test and not dev and not location_in_testset and not location_in_devset):
                                self.example_indices.append((directory_info,int(idx_str)))

                            if not directory_info.silent:
                                location = (info['book'], info['sentence_index'])
                                self.voiced_data_locations[location] = (directory_info,int(idx_str))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        if len(self.example_indices) > 0:
            try:
                sample_mfccs, sample_emg, _, _, _, _ = load_utterance(self.example_indices[0][0].directory, self.example_indices[0][1])
                self.num_speech_features = sample_mfccs.shape[1]
                self.num_features = sample_emg.shape[1]
            except Exception:
                self.num_speech_features = 80
                self.num_features = 8
        else:
            self.num_speech_features = 80
            self.num_features = 8

        self.num_sessions = len(directories)

    def silent_subset(self):
        result = copy(self)
        silent_indices = []
        for example in self.example_indices:
            if example[0].silent:
                silent_indices.append(example)
        result.example_indices = silent_indices
        return result

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[:int(fraction*len(self.example_indices))]
        return result

    def __len__(self):
        return len(self.example_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        if getattr(self, 'emg_file', None) is not None:
            raw_emg = np.load(self.emg_file)
            if raw_emg.ndim == 1:
                raw_emg = raw_emg.reshape(-1, 8)
            if raw_emg.shape[0] == 8 and raw_emg.shape[1] > 8:
                raw_emg = raw_emg.T
            assert raw_emg.shape[1] == 8, f'EMG 應為 (T, 8), 得到 {raw_emg.shape}'

            # ── 從 info.json 讀取自訂前處理參數 ─────────────────────────
            # 支援的 JSON 欄位（皆為選填）：
            #   already_preprocessed : bool   是否已完成 notch/drift/resample (預設 true)
            #   emg_scale            : float  振幅縮放因子 (預設 0.35)
            #   raw_sampling_rate    : float  原始取樣率，用於自動判斷與 resample
            #   target_sampling_rate : float  目標取樣率 (預設 689.06)
            #   notch_freq           : float  市電頻率，50 或 60 Hz (預設 60)
            already_preprocessed = True
            emg_scale = 0.35
            raw_sr = 689.06
            target_sr = 689.06
            notch_freq = 60

            # 嘗試尋找對應的 info.json（支援 trial_XXX_info.json 與 N_info.json 兩種命名）
            info_path = None
            emg_base = os.path.splitext(self.emg_file)[0]  # 去掉 .npy
            if emg_base.endswith('_emg'):
                info_path = emg_base[:-4] + '_info.json'
            elif emg_base.endswith('_inference_689hz'):
                info_path = emg_base[:-len('_inference_689hz')] + '_info.json'

            if info_path and os.path.exists(info_path):
                with open(info_path) as f:
                    file_info = json.load(f)
                already_preprocessed = file_info.get('already_preprocessed', already_preprocessed)
                emg_scale = file_info.get('emg_scale', emg_scale)
                raw_sr = file_info.get('raw_sampling_rate', raw_sr)
                target_sr = file_info.get('target_sampling_rate', target_sr)
                notch_freq = file_info.get('notch_freq', notch_freq)
                # 如果 JSON 沒有明確寫 already_preprocessed，從取樣率自動推斷
                if 'already_preprocessed' not in file_info:
                    already_preprocessed = (abs(raw_sr - target_sr) < 5)

            logging.info(f'EMG preprocessing: already_preprocessed={already_preprocessed}, '
                         f'emg_scale={emg_scale}, raw_sr={raw_sr}, notch_freq={notch_freq}')

            if already_preprocessed:
                # 自錄資料：已完成 notch/drift/resample，只需縮放
                raw_emg = raw_emg * emg_scale
            else:
                # 原始格式：走完整前處理流程（頻率參數皆從 JSON 讀取）
                raw_emg = apply_to_all(notch_harmonics, raw_emg, notch_freq, raw_sr)
                raw_emg = apply_to_all(remove_drift, raw_emg, raw_sr)
                if abs(raw_sr - target_sr) > 1:
                    raw_emg = apply_to_all(subsample, raw_emg, target_sr, raw_sr)

            for c in getattr(FLAGS, 'remove_channels', []):
                raw_emg[:, int(c)] = 0
            raw_emg = raw_emg.astype(np.float32)
            raw_emg = raw_emg / 20.0
            raw_emg = 50.0 * np.tanh(raw_emg / 50.0)

            return {
                'raw_emg': torch.from_numpy(raw_emg),
                'session_ids': torch.zeros(raw_emg.shape[0], dtype=torch.int64),
                'text': '',
                'text_int': torch.zeros(0, dtype=torch.int64),
                'file_label': 0,
                'silent': False,
            }

        directory_info, idx = self.example_indices[i]
        mfccs, emg, text, book_location, phonemes, raw_emg = load_utterance(directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory)
        raw_emg = raw_emg / 20
        raw_emg = 50*np.tanh(raw_emg/50.)

        if not self.no_normalizers:
            mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg)
            emg = 8*np.tanh(emg/8.)

        session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)
        audio_file = f'{directory_info.directory}/{idx}_audio_clean.flac'

        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        result = {'audio_features':torch.from_numpy(mfccs), 'emg':torch.from_numpy(emg), 'text':text, 'text_int': torch.from_numpy(text_int), 'file_label':idx, 'session_ids':torch.from_numpy(session_ids), 'book_location':book_location, 'silent':directory_info.silent, 'raw_emg':torch.from_numpy(raw_emg)}

        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, voiced_emg, _, _, phonemes, _ = load_utterance(voiced_directory.directory, voiced_idx, False, text_align_directory=self.text_align_directory)

            if not self.no_normalizers:
                voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
                voiced_emg = self.emg_norm.normalize(voiced_emg)
                voiced_emg = 8*np.tanh(voiced_emg/8.)

            result['parallel_voiced_audio_features'] = torch.from_numpy(voiced_mfccs)
            result['parallel_voiced_emg'] = torch.from_numpy(voiced_emg)

            audio_file = f'{voiced_directory.directory}/{voiced_idx}_audio_clean.flac'

        result['phonemes'] = torch.from_numpy(phonemes) # either from this example if vocalized or aligned example if silent
        result['audio_file'] = audio_file

        return result

    @staticmethod
    def collate_raw(batch):
        batch_size = len(batch)
        audio_features = []
        audio_feature_lengths = []
        parallel_emg = []
        for ex in batch:
            if ex['silent']:
                audio_features.append(ex['parallel_voiced_audio_features'])
                audio_feature_lengths.append(ex['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(ex['parallel_voiced_emg'])
            else:
                audio_features.append(ex['audio_features'])
                audio_feature_lengths.append(ex['audio_features'].shape[0])
                parallel_emg.append(np.zeros(1))
        phonemes = [ex['phonemes'] for ex in batch]
        emg = [ex['emg'] for ex in batch]
        raw_emg = [ex['raw_emg'] for ex in batch]
        session_ids = [ex['session_ids'] for ex in batch]
        lengths = [ex['emg'].shape[0] for ex in batch]
        silent = [ex['silent'] for ex in batch]
        text_ints = [ex['text_int'] for ex in batch]
        text_lengths = [ex['text_int'].shape[0] for ex in batch]

        result = {'audio_features':audio_features,
                  'audio_feature_lengths':audio_feature_lengths,
                  'emg':emg,
                  'raw_emg':raw_emg,
                  'parallel_voiced_emg':parallel_emg,
                  'phonemes':phonemes,
                  'session_ids':session_ids,
                  'lengths':lengths,
                  'silent':silent,
                  'text_int':text_ints,
                  'text_int_lengths':text_lengths}
        return result

def make_normalizers():
    dataset = EMGDataset(no_normalizers=True)
    mfcc_samples = []
    emg_samples = []
    for d in dataset:
        mfcc_samples.append(d['audio_features'])
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump((mfcc_norm, emg_norm), open(FLAGS.normalizers_file, 'wb'))

if __name__ == '__main__':
    FLAGS(sys.argv)
    if FLAGS.make_normalizers or not os.path.exists(FLAGS.normalizers_file):
        logging.info(f"Generating normalizers -> {FLAGS.normalizers_file}...")
        make_normalizers()
        logging.info(f"✅ Normalizers saved to {FLAGS.normalizers_file}")
    else:
        d = EMGDataset()
        for i in range(min(100, len(d))):
            d[i]
