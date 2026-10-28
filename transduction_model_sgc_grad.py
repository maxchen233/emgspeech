import os
import sys
import csv
import math
import numpy as np
import logging
import subprocess

import soundfile as sf
import tqdm

import torch
import torch.nn.functional as F

from read_emg import EMGDataset, SizeAwareSampler
from architecture import Model
from architecture import PureMambaModel
from align import align_from_distances
from asr_evaluation import evaluate
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length
from vocoder import Vocoder

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 16, 'training batch size')
flags.DEFINE_integer('epochs', 80, 'number of training epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_integer('learning_rate_warmup', 500, 'steps of linear warmup')
flags.DEFINE_string('start_training_from', None, 'start training from this model')
flags.DEFINE_string('teacher_model_path', 'mamba_teacher_model.pt', 'path to the pretrained pure Mamba teacher model')
flags.DEFINE_float('data_size_fraction', 1.0, 'fraction of training data to use')
flags.DEFINE_float('phoneme_loss_weight', 0.5, 'weight of auxiliary phoneme prediction loss')
flags.DEFINE_float('l2', 1e-7, 'weight decay')
flags.DEFINE_string('output_directory', 'output', 'output directory')
flags.DEFINE_integer('grad_log_interval', 10, 'log gradient stats every N steps')
flags.DEFINE_integer('grad_plot_interval', 50, 'write gradient csv every N steps')
flags.DEFINE_float('grad_clip_norm', 5.0, 'gradient clipping max norm')
flags.DEFINE_bool('no_sgc', False, 'disable Smoothed Gradient Compensation path in student model')


def test(model, testset, device):
    model.eval()
    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, collate_fn=testset.collate_raw)
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(phoneme_inventory), len(phoneme_inventory)))
    seq_len = 200
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation', disable=None):
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)
            pred, phoneme_pred = model(None, X_raw, sess)
            loss, phon_acc = dtw_loss(pred, phoneme_pred, batch, True, phoneme_confusion)
            losses.append(loss.item())
            accuracies.append(phon_acc)
    model.train()
    return np.mean(losses), np.mean(accuracies), phoneme_confusion


def save_output(model, datapoint, filename, device, audio_normalizer, vocoder):
    model.eval()
    with torch.no_grad():
        sess = datapoint['session_ids'].to(device=device).unsqueeze(0)
        X_raw = datapoint['raw_emg'].to(dtype=torch.float32, device=device).unsqueeze(0)
        pred, _ = model(None, X_raw, sess)
        y = pred.squeeze(0)
        y = audio_normalizer.inverse(y.cpu()).to(device)
        audio = vocoder(y).cpu().numpy()
    sf.write(filename, audio, 22050)
    model.train()


def get_aligned_prediction(model, datapoint, device, audio_normalizer):
    model.eval()
    with torch.no_grad():
        silent = datapoint['silent']
        sess = datapoint['session_ids'].to(device).unsqueeze(0)
        X_raw = datapoint['raw_emg'].to(device).unsqueeze(0)
        y = datapoint['parallel_voiced_audio_features' if silent else 'audio_features'].to(device).unsqueeze(0)
        pred, _ = model(None, X_raw, sess)
        if silent:
            costs = torch.cdist(pred, y).squeeze(0)
            alignment = align_from_distances(costs.T.detach().cpu().numpy())
            pred_aligned = pred.squeeze(0)[alignment]
        else:
            pred_aligned = pred.squeeze(0)
        pred_aligned = audio_normalizer.inverse(pred_aligned.cpu())
    model.train()
    return pred_aligned


def dtw_loss(predictions, phoneme_predictions, example, phoneme_eval=False, phoneme_confusion=None):
    device = predictions.device
    predictions = decollate_tensor(predictions, example['lengths'])
    phoneme_predictions = decollate_tensor(phoneme_predictions, example['lengths'])
    audio_features = [t.to(device, non_blocking=True) for t in example['audio_features']]
    phoneme_targets = example['phonemes']

    losses = []
    correct_phones = 0
    total_length = 0

    for pred, y, pred_phone, y_phone, silent in zip(
        predictions, audio_features, phoneme_predictions, phoneme_targets, example['silent']
    ):
        y_phone = y_phone.to(device)
        if silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
            costs = dists.squeeze(0)
            pred_phone = F.log_softmax(pred_phone, -1)
            phone_lprobs = pred_phone[:, y_phone]
            costs = costs + FLAGS.phoneme_loss_weight * -phone_lprobs
            alignment = align_from_distances(costs.T.cpu().detach().numpy())
            loss = costs[alignment, range(len(alignment))].sum()
            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == y_phone).sum().item()
                for p, t in zip(pred_phone[alignment].tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1
        else:
            dists = F.pairwise_distance(y, pred)
            phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction='sum')
            loss = dists.sum() + FLAGS.phoneme_loss_weight * phoneme_loss
            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone == y_phone).sum().item()
                for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1
        losses.append(loss)
        total_length += y.size(0)

    return sum(losses) / total_length, correct_phones / total_length


class GradientMonitorV2:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.tensor_stats = {}
        self.step_rows = []
        self.param_rows = []
        self.tensor_rows = []
        self._handles = []
        self.sgc_param_rows = []

    def reset_tensors(self):
        self.tensor_stats = {}

    def register_tensor(self, name, tensor):
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        t = tensor.detach().float()
        self.tensor_stats[name] = {
            'act_mean': t.mean().item(),
            'act_std': t.std().item() if t.numel() > 1 else 0.0,
            'act_abs_max': t.abs().max().item(),
            'act_nonzero_ratio': t.ne(0).float().mean().item(),
        }
        if tensor.requires_grad:
            tensor.retain_grad()

            def _hook(grad, tensor_name=name):
                g = grad.detach().float()
                row = self.tensor_stats.setdefault(tensor_name, {})
                row.update({
                    'grad_mean': g.mean().item(),
                    'grad_std': g.std().item() if g.numel() > 1 else 0.0,
                    'grad_norm': g.norm().item(),
                    'grad_abs_max': g.abs().max().item(),
                    'grad_nonzero_ratio': g.ne(0).float().mean().item(),
                })
            self._handles.append(tensor.register_hook(_hook))

    def register_named_module_outputs(self, model, module_keywords=None):
        if module_keywords is None:
            module_keywords = ['sgc', 'spike', 'mamba']
        for name, module in model.named_modules():
            lname = name.lower()
            if not name:
                continue
            if not any(k in lname for k in module_keywords):
                continue

            def _forward_hook(mod, inp, out, module_name=name):
                if isinstance(out, torch.Tensor):
                    self.register_tensor(f'module/{module_name}', out)
                elif isinstance(out, (tuple, list)):
                    for idx, item in enumerate(out):
                        if isinstance(item, torch.Tensor):
                            self.register_tensor(f'module/{module_name}[{idx}]', item)
            self._handles.append(module.register_forward_hook(_forward_hook))

    def collect_param_grads(self, model, step, stage='pre_clip'):
        rows = []
        total_sq = 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            weight = p.detach().float()
            grad_norm = grad.norm().item()
            weight_norm = weight.norm().item()
            row = {
                'step': step,
                'stage': stage,
                'name': name,
                'grad_norm': grad_norm,
                'weight_norm': weight_norm,
                'grad_to_weight': grad_norm / (weight_norm + 1e-12),
                'grad_mean': grad.mean().item(),
                'grad_std': grad.std().item() if grad.numel() > 1 else 0.0,
                'grad_abs_max': grad.abs().max().item(),
                'grad_nonzero_ratio': grad.ne(0).float().mean().item(),
            }
            rows.append(row)
            total_sq += grad_norm ** 2
        self.param_rows.extend(rows)
        return rows, math.sqrt(total_sq)

    def collect_sgc_param_grads(self, model, step, stage='pre_clip'):
        rows = []
        for name, p in model.named_parameters():
            if 'mamba_sgc' not in name.lower():
                continue

            row = {
                'step': step,
                'stage': stage,
                'name': name,
                'requires_grad': p.requires_grad,
                'has_grad': p.grad is not None,
            }

            if p.grad is not None:
                grad = p.grad.detach().float()
                weight = p.detach().float()
                row.update({
                    'grad_norm': grad.norm().item(),
                    'weight_norm': weight.norm().item(),
                    'grad_to_weight': grad.norm().item() / (weight.norm().item() + 1e-12),
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item() if grad.numel() > 1 else 0.0,
                    'grad_abs_max': grad.abs().max().item(),
                    'grad_nonzero_ratio': grad.ne(0).float().mean().item(),
                })
            else:
                row.update({
                    'grad_norm': None,
                    'weight_norm': p.detach().float().norm().item(),
                    'grad_to_weight': None,
                    'grad_mean': None,
                    'grad_std': None,
                    'grad_abs_max': None,
                    'grad_nonzero_ratio': None,
                })

            rows.append(row)
        self.sgc_param_rows.extend(rows)
        return rows

    def collect_tensors(self, step):
        rows = []
        for name, stats in self.tensor_stats.items():
            row = {'step': step, 'name': name}
            row.update(stats)
            rows.append(row)
        self.tensor_rows.extend(rows)
        return rows

    def record_step(self, step, payload):
        row = {'step': step}
        row.update(payload)
        self.step_rows.append(row)

    def log_topk(self, rows, step, k=5):
        if not rows:
            return
        top = sorted(rows, key=lambda x: x['grad_norm'], reverse=True)[:k]
        msg = ' | '.join([f"{r['name']}:{r['grad_norm']:.3e}" for r in top])
        logging.info(f'[grad][step={step}] top{k} grad_norm {msg}')

    def flush_csv(self):
        self._write_csv(os.path.join(self.save_dir, 'step_metrics.csv'), self.step_rows)
        self._write_csv(os.path.join(self.save_dir, 'param_grad_metrics.csv'), self.param_rows)
        self._write_csv(os.path.join(self.save_dir, 'tensor_grad_metrics.csv'), self.tensor_rows)
        self._write_csv(os.path.join(self.save_dir, 'sgc_param_grad_metrics.csv'), self.sgc_param_rows)

    def _write_csv(self, path, rows):
        if not rows:
            return
        keys = sorted(set().union(*[r.keys() for r in rows]))
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def close(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


def train_model(trainset, devset, device, save_sound_outputs=True):
    monitor = GradientMonitorV2(save_dir=os.path.join(FLAGS.output_directory, 'grad_monitor_v2'))
    vocoder = Vocoder()

    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = trainset.subset(FLAGS.data_size_fraction)

    dataloader = torch.utils.data.DataLoader(
        training_subset,
        pin_memory=(device == 'cuda'),
        collate_fn=devset.collate_raw,
        num_workers=0,
        batch_sampler=SizeAwareSampler(training_subset, 64000),
    )

    n_phones = len(phoneme_inventory)
    model_student = Model(devset.num_features, devset.num_speech_features, n_phones, no_sgc=FLAGS.no_sgc).to(device)
    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model_student.load_state_dict(state_dict, strict=True)

    model_teacher = PureMambaModel(devset.num_features, devset.num_speech_features, n_phones).to(device)
    if FLAGS.teacher_model_path is not None:
        logging.info(f'Loading teacher weights from: {FLAGS.teacher_model_path}')
        teacher_state_dict = torch.load(FLAGS.teacher_model_path, map_location=device, weights_only=True)
        model_teacher.load_state_dict(teacher_state_dict, strict=True)
    else:
        logging.warning('Warning: No teacher path provided! Self-distillation running in random state.')
    model_teacher.eval()
    for param in model_teacher.parameters():
        param.requires_grad = False

    monitor.register_named_module_outputs(model_student, module_keywords=['sgc', 'spike', 'mamba'])

    optim = torch.optim.AdamW(model_student.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.5, patience=FLAGS.learning_rate_patience)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration * target_lr / FLAGS.learning_rate_warmup)

    seq_len = 200
    batch_idx = 0

    for epoch_idx in range(FLAGS.epochs):
        losses = []
        pbar = tqdm.tqdm(dataloader, f'Train epoch {epoch_idx + 1}', disable=None)
        for batch in pbar:
            monitor.reset_tensors()
            optim.zero_grad(set_to_none=True)
            schedule_lr(batch_idx)

            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len * 8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)

            pred_student, phone_student = model_student(None, X_raw, sess)
            monitor.register_tensor('continuous/pred_student', pred_student)
            monitor.register_tensor('continuous/phone_student', phone_student)

            last_spike = getattr(model_student, 'last_spike_tensor', None)
            if last_spike is not None:
                monitor.register_tensor('spike/last_spike_tensor', last_spike)
                actual_spike_rate = last_spike.detach().ne(0).float().mean().item()
            else:
                actual_spike_rate = 0.0

            with torch.no_grad():
                pred_teacher, _ = model_teacher(None, X_raw, sess)

            base_loss, _ = dtw_loss(pred_student, phone_student, batch)
            loss_kd = F.mse_loss(pred_student, pred_teacher)
            loss_hidden = sum(getattr(layer, 'current_hidden_loss', 0.0) for layer in getattr(model_student, 'mamba_layers', []))
            if not torch.is_tensor(loss_hidden):
                loss_hidden = torch.tensor(float(loss_hidden), device=device)

            total_loss = base_loss + 1.0 * loss_kd + 0.5 * loss_hidden
            total_loss.backward()
            losses.append(total_loss.item())

            sgc_pre_rows = monitor.collect_sgc_param_grads(
                model_student, step=batch_idx, stage='pre_clip')
            for r in sgc_pre_rows:
                logging.info(
                    f"[sgc][step={batch_idx}][pre_clip] {r['name']} "
                    f"has_grad={r['has_grad']} grad_norm={r['grad_norm']}")

            pre_rows, pre_clip_norm = monitor.collect_param_grads(model_student, step=batch_idx, stage='pre_clip')
            monitor.collect_tensors(step=batch_idx)

            clip_return = torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=FLAGS.grad_clip_norm)

            sgc_post_rows = monitor.collect_sgc_param_grads(
                model_student, step=batch_idx, stage='post_clip'
            )
            
            for r in sgc_post_rows:
                logging.info(
                    f"[sgc][step={batch_idx}][post_clip] {r['name']} "
                    f"has_grad={r['has_grad']} grad_norm={r['grad_norm']}"
                )

            _, post_clip_norm = monitor.collect_param_grads(model_student, step=batch_idx, stage='post_clip')

            monitor.record_step(batch_idx, {
                'epoch': epoch_idx + 1,
                'lr': optim.param_groups[0]['lr'],
                'loss_total': total_loss.item(),
                'loss_base': base_loss.item(),
                'loss_kd': loss_kd.item(),
                'loss_hidden': loss_hidden.item(),
                'spike_rate': actual_spike_rate,
                'grad_global_pre_clip': pre_clip_norm,
                'grad_global_pre_clip_return': float(clip_return),
                'grad_global_post_clip': post_clip_norm,
            })

            if batch_idx % FLAGS.grad_log_interval == 0:
                monitor.log_topk(pre_rows, step=batch_idx, k=5)
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.3f}',
                    'spike': f'{actual_spike_rate:.2%}',
                    'gpre': f'{pre_clip_norm:.2f}',
                    'gpost': f'{post_clip_norm:.2f}',
                })

            if batch_idx % FLAGS.grad_plot_interval == 0:
                monitor.flush_csv()

            optim.step()
            batch_idx += 1

        train_loss = float(np.mean(losses)) if losses else float('nan')
        val, phoneme_acc, _ = test(model_student, devset, device)
        lr_sched.step(val)
        logging.info(
            f'finished epoch {epoch_idx + 1} - validation loss: {val:.4f} '
            f'training loss: {train_loss:.4f} phoneme accuracy: {phoneme_acc * 100:.2f}'
        )
        torch.save(model_student.state_dict(), os.path.join(FLAGS.output_directory, 'model.pt'))
        monitor.flush_csv()

        if getattr(FLAGS, 'hifigan_checkpoint', None) is not None:
            save_output(
                model_student,
                devset[0],
                os.path.join(FLAGS.output_directory, f'epoch_{epoch_idx}_output.wav'),
                device,
                devset.mfcc_norm,
                vocoder,
            )

    if save_sound_outputs:
        for i, datapoint in enumerate(devset):
            save_output(
                model_student,
                datapoint,
                os.path.join(FLAGS.output_directory, f'example_output_{i}.wav'),
                device,
                devset.mfcc_norm,
                vocoder,
            )
        evaluate(devset, FLAGS.output_directory)

    monitor.flush_csv()
    monitor.close()
    return model_student


def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
        ],
        level=logging.INFO,
        format='%(message)s'
    )

    logging.info(subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(sys.argv)

    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d', len(trainset), len(devset))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(trainset, devset, device, save_sound_outputs=(getattr(FLAGS, 'hifigan_checkpoint', None) is not None))


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()