"""SNN transduction model training script.

Trains an EMG-to-speech transduction model whose sequence encoder is a
multi-layer Spiking Neural Network (SNN).

Learning strategy
-----------------
Primary  – BPTT with surrogate gradients (ATan or triangular).
             Gradients flow through the LIF spike function via the surrogate.
Optional – STDP local weight update applied every forward pass when
             ``--snn_use_stdp`` is set.
Optional – SALSA spike-rate regularisation loss (penalises deviation from a
             target firing rate) added to the main loss when
             ``--snn_salsa_weight > 0``.

Usage example
-------------
    python snn_transduction_model.py \\
        --output_directory snn_output \\
        --snn_n_layers 4 \\
        --snn_surrogate atan \\
        --snn_use_stdp \\
        --snn_salsa_weight 0.01 \\
        --snn_use_checkpoint \\
        --grad_accum_steps 4

VRAM reduction flags
--------------------
    --snn_use_checkpoint   gradient checkpointing per SNN block
    --grad_accum_steps N   accumulate N mini-batches before each optimizer step
"""
from __future__ import annotations

import os
import sys
import logging
import subprocess

import numpy as np
import soundfile as sf
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from read_emg import EMGDataset, SizeAwareSampler
from align import align_from_distances
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
from architecture import ResBlock

from snn_model import SNNEncoder, salsa_spike_rate_loss

try:
    from vocoder import Vocoder
    _HAS_VOCODER = True
except ImportError:
    _HAS_VOCODER = False

from absl import flags

FLAGS = flags.FLAGS

# ── SNN hyper-parameters ────────────────────────────────────────────────────
flags.DEFINE_integer('snn_n_layers',   4,     'number of SNN encoder layers')
flags.DEFINE_float  ('snn_hidden_scale', 2.0, 'hidden-layer width relative to d_model')
flags.DEFINE_float  ('snn_threshold',  1.0,   'LIF firing threshold')
flags.DEFINE_float  ('snn_decay',      0.9,   'LIF membrane decay factor')
flags.DEFINE_string ('snn_surrogate',  'atan','surrogate gradient: "atan" or "triangular"')
flags.DEFINE_float  ('snn_dropout',    0.2,   'dropout after output projection')
flags.DEFINE_bool   ('snn_use_stdp',   False, 'enable STDP local weight updates')
flags.DEFINE_float  ('snn_salsa_weight', 0.0, 'weight for SALSA spike-rate regularisation loss')
flags.DEFINE_float  ('snn_target_rate', 0.1,  'target mean firing rate for SALSA')
flags.DEFINE_bool   ('snn_use_checkpoint', False, 'gradient checkpointing in SNN blocks')

# ── General training flags ───────────────────────────────────────────────────
flags.DEFINE_integer('batch_size',              8,    'training batch size')
flags.DEFINE_integer('epochs',                  80,   'number of training epochs')
flags.DEFINE_float  ('learning_rate',           1e-3, 'learning rate')
flags.DEFINE_integer('learning_rate_patience',  5,    'LR-reduction patience')
flags.DEFINE_integer('learning_rate_warmup',    500,  'warmup steps')
flags.DEFINE_string ('start_training_from',     None, 'path to pre-trained weights')
flags.DEFINE_float  ('data_size_fraction',      1.0,  'fraction of training data')
flags.DEFINE_float  ('phoneme_loss_weight',     0.5,  'auxiliary phoneme loss weight')
flags.DEFINE_float  ('l2',                      1e-7, 'weight decay')
flags.DEFINE_string ('output_directory',        'snn_output', 'output directory')
flags.DEFINE_integer('grad_accum_steps',        1,
                     'gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)')
flags.DEFINE_bool   ('pin_memory',              True, 'pin DataLoader memory')
flags.DEFINE_integer('model_size',              256,  'hidden dimension')
flags.DEFINE_float  ('dropout',                 0.2,  'dropout (passed to SNN encoder)')


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SNNTransductionModel(nn.Module):
    """EMG-to-speech transduction model with an SNN sequence encoder.

    Architecture
    ------------
    1. Three residual Conv1D blocks for local feature extraction.
    2. A linear projection to d_model.
    3. Multi-layer SNNEncoder for temporal modelling.
    4. Two output heads: mel-spectrogram prediction + phoneme prediction.
    """

    def __init__(self, num_features: int, num_outs: int, num_aux_outs: int):
        super().__init__()
        d_model = FLAGS.model_size

        # ── same conv front-end as the Mamba / Transformer models ───────────
        self.conv_blocks = nn.Sequential(
            ResBlock(8, d_model, 2),
            ResBlock(d_model, d_model, 2),
            ResBlock(d_model, d_model, 2),
        )
        self.w_raw_in = nn.Linear(d_model, d_model)

        # ── SNN encoder ──────────────────────────────────────────────────────
        self.sequence_model = SNNEncoder(
            d_model       = d_model,
            n_layers      = FLAGS.snn_n_layers,
            hidden_scale  = FLAGS.snn_hidden_scale,
            threshold     = FLAGS.snn_threshold,
            decay         = FLAGS.snn_decay,
            surrogate     = FLAGS.snn_surrogate,
            dropout       = FLAGS.snn_dropout,
            use_stdp      = FLAGS.snn_use_stdp,
            use_checkpoint= FLAGS.snn_use_checkpoint,
        )

        # ── output heads ─────────────────────────────────────────────────────
        self.w_out  = nn.Linear(d_model, num_outs)
        self.w_aux  = nn.Linear(d_model, num_aux_outs)

    def forward(
        self,
        x_feat: torch.Tensor,
        x_raw: torch.Tensor,
        session_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_feat      : (batch, time, electrode_features)  – unused here
            x_raw       : (batch, time, 8)                   – raw EMG channels
            session_ids : (batch, time)                      – unused here

        Returns:
            pred         : (batch, time, num_outs)
            phoneme_pred : (batch, time, num_aux_outs)
        """
        import random
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]
                x_raw[:, -r:, :] = 0

        x = x_raw.transpose(1, 2)      # (batch, 8, time)
        x = self.conv_blocks(x)         # (batch, d_model, time')
        x = x.transpose(1, 2)          # (batch, time', d_model)
        x = self.w_raw_in(x)

        x = x.transpose(0, 1)          # (time', batch, d_model)
        x = self.sequence_model(x)      # (time', batch, d_model)
        x = x.transpose(0, 1)          # (batch, time', d_model)

        return self.w_out(x), self.w_aux(x)


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers  (identical to transduction_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def dtw_loss(
    predictions: torch.Tensor,
    phoneme_predictions: torch.Tensor,
    example: dict,
    phoneme_eval: bool = False,
    phoneme_confusion: np.ndarray | None = None,
) -> tuple[torch.Tensor, float]:
    device = predictions.device

    predictions        = decollate_tensor(predictions, example['lengths'])
    phoneme_predictions = decollate_tensor(phoneme_predictions, example['lengths'])

    audio_features  = [t.to(device, non_blocking=True) for t in example['audio_features']]
    phoneme_targets = example['phonemes']

    losses: list[torch.Tensor] = []
    correct_phones = 0
    total_length   = 0

    for pred, y, pred_phone, y_phone, silent in zip(
        predictions, audio_features, phoneme_predictions, phoneme_targets, example['silent']
    ):
        assert len(pred.size()) == 2 and len(y.size()) == 2
        y_phone = y_phone.to(device)

        if silent:
            dists      = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
            costs      = dists.squeeze(0)
            pred_phone = F.log_softmax(pred_phone, -1)
            phone_lprobs = pred_phone[:, y_phone]
            costs      = costs + FLAGS.phoneme_loss_weight * -phone_lprobs
            alignment  = align_from_distances(costs.T.cpu().detach().numpy())
            loss       = costs[alignment, range(len(alignment))].sum()

            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == y_phone).sum().item()
                if phoneme_confusion is not None:
                    for p, t in zip(pred_phone[alignment].tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1
        else:
            assert y.size(0) == pred.size(0)
            dists        = F.pairwise_distance(y, pred)
            phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction='sum')
            loss         = dists.sum() + FLAGS.phoneme_loss_weight * phoneme_loss

            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone == y_phone).sum().item()
                if phoneme_confusion is not None:
                    for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1

        losses.append(loss)
        total_length += y.size(0)

    return sum(losses) / total_length, correct_phones / total_length


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def test(model: SNNTransductionModel, testset, device: str):
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=32, collate_fn=testset.collate_raw
    )
    losses: list[float] = []
    accuracies: list[float] = []
    phoneme_confusion = np.zeros((len(phoneme_inventory), len(phoneme_inventory)))
    seq_len = 200

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation', disable=None):
            X     = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
            sess  = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            with amp.autocast():
                pred, phoneme_pred = model(X, X_raw, sess)
                loss, phon_acc = dtw_loss(pred, phoneme_pred, batch, True, phoneme_confusion)
            losses.append(loss.item())
            accuracies.append(phon_acc)

    model.train()
    return np.mean(losses), np.mean(accuracies), phoneme_confusion


def save_output(model, datapoint, filename, device, audio_normalizer, vocoder):
    model.eval()
    with torch.no_grad():
        sess  = datapoint['session_ids'].to(device=device).unsqueeze(0)
        X     = datapoint['emg'].to(dtype=torch.float32, device=device).unsqueeze(0)
        X_raw = datapoint['raw_emg'].to(dtype=torch.float32, device=device).unsqueeze(0)
        with amp.autocast():
            pred, _ = model(X, X_raw, sess)
        y     = pred.squeeze(0)
        y     = audio_normalizer.inverse(y.cpu()).to(device)
        audio = vocoder(y).cpu().numpy()
    sf.write(filename, audio, 22050)
    model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(trainset, devset, device: str, save_sound_outputs: bool = True):
    n_epochs = FLAGS.epochs

    training_subset = (
        trainset if FLAGS.data_size_fraction >= 1 else trainset.subset(FLAGS.data_size_fraction)
    )
    dataloader = torch.utils.data.DataLoader(
        training_subset,
        pin_memory=(device == 'cuda' and FLAGS.pin_memory),
        collate_fn=devset.collate_raw,
        num_workers=0,
        batch_sampler=SizeAwareSampler(training_subset, 256000),
    )

    n_phones = len(phoneme_inventory)
    model = SNNTransductionModel(
        devset.num_features, devset.num_speech_features, n_phones
    ).to(device)

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    vocoder = None
    if save_sound_outputs:
        if _HAS_VOCODER:
            vocoder = Vocoder()
        else:
            logging.warning('Vocoder not available – audio output disabled.')
            save_sound_outputs = False

    optim    = torch.optim.AdamW(model.parameters(), weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', 0.5, patience=FLAGS.learning_rate_patience
    )
    scaler   = amp.GradScaler()

    target_lr  = FLAGS.learning_rate
    grad_accum = max(1, FLAGS.grad_accum_steps)

    def set_lr(new_lr: float) -> None:
        for pg in optim.param_groups:
            pg['lr'] = new_lr

    def schedule_lr(iteration: int) -> None:
        step = iteration + 1
        if step <= FLAGS.learning_rate_warmup:
            set_lr(step * target_lr / FLAGS.learning_rate_warmup)

    seq_len   = 200
    batch_idx = 0

    for epoch_idx in range(n_epochs):
        losses: list[float] = []
        optim.zero_grad()

        for step_idx, batch in enumerate(tqdm.tqdm(dataloader, 'Train step', disable=None)):
            schedule_lr(batch_idx)

            X     = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']],     seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
            sess  = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            with amp.autocast():
                pred, phoneme_pred = model(X, X_raw, sess)
                task_loss, _ = dtw_loss(pred, phoneme_pred, batch)

                # ── SALSA spike-rate regularisation ──────────────────────────
                salsa_loss = torch.tensor(0.0, device=device)
                if FLAGS.snn_salsa_weight > 0:
                    # Use actual spike tensors collected by SNNEncoder.forward()
                    spike_tensors = model.sequence_model.last_spikes
                    if spike_tensors:
                        all_spikes = torch.cat(spike_tensors, dim=-1)
                        salsa_loss = salsa_spike_rate_loss(all_spikes, FLAGS.snn_target_rate)

                loss = task_loss + FLAGS.snn_salsa_weight * salsa_loss
                loss = loss / grad_accum

            losses.append(loss.item() * grad_accum)
            scaler.scale(loss).backward()

            if (step_idx + 1) % grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            batch_idx += 1

        # flush remaining gradients
        if (step_idx + 1) % grad_accum != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        train_loss = np.mean(losses)
        torch.cuda.empty_cache()
        val, phoneme_acc, _ = test(model, devset, device)
        lr_sched.step(val)
        logging.info(
            f'epoch {epoch_idx+1:3d} | val {val:.4f} | train {train_loss:.4f} | '
            f'phone_acc {phoneme_acc*100:.2f}%'
        )
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, 'snn_model.pt'))

        if save_sound_outputs:
            save_output(
                model, devset[0],
                os.path.join(FLAGS.output_directory, f'epoch_{epoch_idx}_output.wav'),
                device, devset.mfcc_norm, vocoder,
            )

    if save_sound_outputs:
        for i, datapoint in enumerate(devset):
            save_output(
                model, datapoint,
                os.path.join(FLAGS.output_directory, f'example_output_{i}.wav'),
                device, devset.mfcc_norm, vocoder,
            )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format='%(message)s',
    )

    logging.info(subprocess.run(['git', 'rev-parse', 'HEAD'],
                                stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(subprocess.run(['git', 'diff'],
                                stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(sys.argv)

    trainset = EMGDataset(dev=False, test=False)
    devset   = EMGDataset(dev=True)
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d / %d', len(trainset), len(devset))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('using device: %s', device)

    train_model(trainset, devset, device, save_sound_outputs=_HAS_VOCODER)


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
