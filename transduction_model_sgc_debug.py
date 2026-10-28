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
flags.DEFINE_integer('debug_max_train_steps', -1, 'stop training after this many global steps; -1 disables')
flags.DEFINE_integer('debug_max_eval_batches', -1, 'limit validation to this many batches; -1 disables')
flags.DEFINE_bool('debug_skip_validation', False, 'skip validation entirely for faster debugging')
flags.DEFINE_bool('debug_cuda_launch_blocking', False, 'set CUDA_LAUNCH_BLOCKING=1 before model construction')
flags.DEFINE_bool('debug_flush_each_step', False, 'flush gradient csv after every step for crash-safe debugging')

def debug_tensor(tag, x):
    if x is None:
        print(f'[debug] {tag}: None')
        return

    if not isinstance(x, torch.Tensor):
        print(f'[debug] {tag}: type={type(x)}')
        return

    finite = True
    try:
        finite = torch.isfinite(x).all().item() if x.is_floating_point() or x.is_complex() else True
    except Exception:
        finite = True

    print(
        f'[debug] {tag}: '
        f'shape={tuple(x.shape)} device={x.device} dtype={x.dtype} finite={finite}'
    )

    if x.numel() > 0:
        xd = x.detach()
        if x.is_floating_point() or x.is_complex():
            print(
                f'[debug] {tag} stats: '
                f'min={float(xd.min())} max={float(xd.max())} mean={float(xd.mean())}'
            )
        else:
            print(
                f'[debug] {tag} stats: '
                f'min={int(xd.min())} max={int(xd.max())} '
                f'unique={xd.unique().numel()}'
            )


def dump_batch(batch, prefix='batch'):
    print(f'[debug] dumping {prefix}')
    print(f'[debug] {prefix} keys: {list(batch.keys())}')
    for k, v in batch.items():
        if isinstance(v, list):
            print(f'[debug] {prefix}[{k}] -> list len={len(v)}')
            if len(v) > 0:
                first = v[0]
                if isinstance(first, torch.Tensor):
                    print(
                        f'[debug] {prefix}[{k}][0]: '
                        f'shape={tuple(first.shape)} dtype={first.dtype} device={first.device}'
                    )
                else:
                    print(f'[debug] {prefix}[{k}][0] type={type(first)}')
        elif isinstance(v, torch.Tensor):
            print(
                f'[debug] {prefix}[{k}]: '
                f'shape={tuple(v.shape)} dtype={v.dtype} device={v.device}'
            )
        else:
            print(f'[debug] {prefix}[{k}] type={type(v)} value={v}')


def test(model, testset, device):
    max_eval_batches = getattr(FLAGS, 'debug_max_eval_batches', -1)
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=32,
        collate_fn=testset.collate_raw
    )
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(phoneme_inventory), len(phoneme_inventory)))
    seq_len = 200

    with torch.no_grad():
        for eval_batch_idx, batch in enumerate(tqdm.tqdm(dataloader, 'Validation', disable=None)):
            if max_eval_batches >= 0 and eval_batch_idx >= max_eval_batches:
                break

            try:
                print(f'[debug] ===== validation batch {eval_batch_idx} =====')
                dump_batch(batch, prefix=f'val_batch_{eval_batch_idx}')

                X_raw = combine_fixed_length(
                    [t.to(device, non_blocking=True) for t in batch['raw_emg']],
                    seq_len * 8
                )
                sess = combine_fixed_length(
                    [t.to(device, non_blocking=True) for t in batch['session_ids']],
                    seq_len
                )

                debug_tensor('eval/X_raw', X_raw)
                debug_tensor('eval/sess', sess)

                pred, phoneme_pred = model(None, X_raw, sess)

                debug_tensor('eval/pred', pred)
                debug_tensor('eval/phoneme_pred', phoneme_pred)

                print(
                    '[debug] before dtw_loss:',
                    pred.shape,
                    phoneme_pred.shape,
                    torch.isfinite(pred).all().item(),
                    torch.isfinite(phoneme_pred).all().item()
                )

                loss, phon_acc = dtw_loss(pred, phoneme_pred, batch, True, phoneme_confusion)

                print('[debug] after dtw_loss:', float(loss), phon_acc)

                if not torch.isfinite(loss):
                    raise RuntimeError(f'Validation loss is non-finite at eval batch {eval_batch_idx}')

                losses.append(loss.item())
                accuracies.append(phon_acc)

            except Exception:
                print(f'[debug] validation crashed at eval batch {eval_batch_idx}')
                dump_batch(batch, prefix=f'val_batch_{eval_batch_idx}_crash')
                raise

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

    for ex_idx, (pred, y, pred_phone, y_phone, silent) in enumerate(
        zip(predictions, audio_features, phoneme_predictions, phoneme_targets, example['silent'])
    ):
        try:
            print(f'[debug] ----- dtw example {ex_idx} silent={silent} -----')
            debug_tensor('dtw/pred', pred)
            debug_tensor('dtw/y', y)
            debug_tensor('dtw/pred_phone_raw', pred_phone)

            y_phone = y_phone.to(device)
            debug_tensor('dtw/y_phone', y_phone)

            if silent:
                dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
                costs = dists.squeeze(0)
                debug_tensor('dtw/dists', costs)

                pred_phone = F.log_softmax(pred_phone, -1)
                debug_tensor('dtw/pred_phone_logsoftmax', pred_phone)

                phone_lprobs = pred_phone[:, y_phone]
                debug_tensor('dtw/phone_lprobs', phone_lprobs)

                costs = costs + FLAGS.phoneme_loss_weight * -phone_lprobs
                print('[debug] costs shape:', costs.shape)
                debug_tensor('dtw/costs_total', costs)

                if not torch.isfinite(costs).all():
                    raise RuntimeError(f'Non-finite costs at example {ex_idx}')

                alignment = align_from_distances(costs.T.cpu().detach().numpy())
                print('[debug] alignment type:', type(alignment), 'len:', len(alignment))

                if len(alignment) != y.shape[0]:
                    raise RuntimeError(
                        f'Alignment length mismatch at example {ex_idx}: '
                        f'len(alignment)={len(alignment)} vs target_len={y.shape[0]}'
                    )

                alignment_t = torch.as_tensor(alignment, device=device, dtype=torch.long)

                if alignment_t.numel() > 0:
                    amin = alignment_t.min().item()
                    amax = alignment_t.max().item()
                    print(f'[debug] alignment min/max: {amin} {amax} pred_len={pred.shape[0]}')
                    if amin < 0 or amax >= pred.shape[0]:
                        raise RuntimeError(
                            f'Alignment out of bounds at example {ex_idx}: '
                            f'min={amin} max={amax} pred_len={pred.shape[0]}'
                        )

                loss = costs[alignment_t, torch.arange(len(alignment), device=device)].sum()
                print('[debug] loss done', float(loss))

                if phoneme_eval:
                    pred_phone_ids = pred_phone.argmax(-1)
                    correct_phones += (pred_phone_ids[alignment_t] == y_phone).sum().item()
                    for p, t in zip(pred_phone_ids[alignment_t].tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1

            else:
                dists = F.pairwise_distance(y, pred)
                debug_tensor('dtw/pairwise_distance', dists)

                phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction='sum')
                if not torch.isfinite(phoneme_loss):
                    raise RuntimeError(f'Non-finite phoneme_loss at example {ex_idx}')

                loss = dists.sum() + FLAGS.phoneme_loss_weight * phoneme_loss
                print('[debug] non-silent loss done', float(loss))

                if phoneme_eval:
                    pred_phone_ids = pred_phone.argmax(-1)
                    correct_phones += (pred_phone_ids == y_phone).sum().item()
                    for p, t in zip(pred_phone_ids.tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1

            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite final loss at example {ex_idx}')

            losses.append(loss)
            total_length += y.size(0)

        except Exception:
            print(f'[debug] dtw_loss crashed at example {ex_idx}')
            print(
                f'[debug] pred_len={pred.shape[0]} '
                f'target_len={y.shape[0]} '
                f'pred_phone_shape={tuple(pred_phone.shape)} '
                f'y_phone_len={y_phone.shape[0]} '
                f'silent={silent}'
            )
            raise

    total_loss = sum(losses) / total_length
    return total_loss, correct_phones / total_length


class GradientMonitorV2:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.tensor_stats = {}
        self.step_rows = []
        self.param_rows = []
        self.tensor_rows = []
        self._handles = []

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
    monitor = None  # GradientMonitorV2(save_dir=os.path.join(FLAGS.output_directory, 'grad_monitor_v2'))
    vocoder = None
    if save_sound_outputs:
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

    model_student = Model(
        devset.num_features,
        devset.num_speech_features,
        n_phones,
        no_sgc=FLAGS.no_sgc
    ).to(device)

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model_student.load_state_dict(state_dict, strict=True)

    model_teacher = PureMambaModel(
        devset.num_features,
        devset.num_speech_features,
        n_phones
    ).to(device)

    if FLAGS.teacher_model_path is not None:
        logging.info(f'Loading teacher weights from: {FLAGS.teacher_model_path}')
        teacher_state_dict = torch.load(
            FLAGS.teacher_model_path,
            map_location=device,
            weights_only=True
        )
        model_teacher.load_state_dict(teacher_state_dict, strict=True)
    else:
        logging.warning('Warning: No teacher path provided! Self-distillation running in random state.')

    model_teacher.eval()
    for param in model_teacher.parameters():
        param.requires_grad = False

    optim = torch.optim.AdamW(
        model_student.parameters(),
        lr=FLAGS.learning_rate,
        weight_decay=FLAGS.l2
    )
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        'min',
        factor=0.5,
        patience=FLAGS.learning_rate_patience
    )

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
    stop_training_early = False

    for epoch_idx in range(FLAGS.epochs):
        losses = []
        pbar = tqdm.tqdm(dataloader, f'Train epoch {epoch_idx + 1}', disable=None)

        for batch in pbar:
            if FLAGS.debug_max_train_steps > 0 and batch_idx >= FLAGS.debug_max_train_steps:
                logging.info(
                    f'[debug] reached debug_max_train_steps={FLAGS.debug_max_train_steps}; '
                    f'stopping training loop before step {batch_idx}'
                )
                stop_training_early = True
                break

            try:
                print(f'[debug] ===== train batch {batch_idx} =====')
                dump_batch(batch, prefix=f'train_batch_{batch_idx}')

                optim.zero_grad(set_to_none=True)
                schedule_lr(batch_idx)

                X_raw = combine_fixed_length(
                    [t.to(device, non_blocking=True) for t in batch['raw_emg']],
                    seq_len * 8
                )
                sess = combine_fixed_length(
                    [t.to(device, non_blocking=True) for t in batch['session_ids']],
                    seq_len
                )

                debug_tensor('train/X_raw', X_raw)
                debug_tensor('train/sess', sess)

                pred_student, phone_student = model_student(None, X_raw, sess)
                debug_tensor('train/pred_student', pred_student)
                debug_tensor('train/phone_student', phone_student)

                last_spike = getattr(model_student, 'last_spike_tensor', None)
                if last_spike is not None:
                    actual_spike_rate = last_spike.detach().ne(0).float().mean().item()
                else:
                    actual_spike_rate = 0.0

                with torch.no_grad():
                    pred_teacher, _ = model_teacher(None, X_raw, sess)
                    debug_tensor('train/pred_teacher', pred_teacher)

                base_loss, _ = dtw_loss(pred_student, phone_student, batch)
                loss_kd = F.mse_loss(pred_student, pred_teacher)
                loss_hidden = sum(
                    getattr(layer, 'current_hidden_loss', 0.0)
                    for layer in getattr(model_student, 'mamba_layers', [])
                )
                if not torch.is_tensor(loss_hidden):
                    loss_hidden = torch.tensor(float(loss_hidden), device=device)

                total_loss = base_loss + 1.0 * loss_kd + 0.5 * loss_hidden

                print(
                    '[debug] losses:',
                    'base=', float(base_loss),
                    'kd=', float(loss_kd),
                    'hidden=', float(loss_hidden),
                    'total=', float(total_loss),
                )

                if not torch.isfinite(total_loss):
                    raise RuntimeError(f'Non-finite total_loss at train batch {batch_idx}')

                total_loss.backward()
                losses.append(total_loss.item())

                clip_return = torch.nn.utils.clip_grad_norm_(
                    model_student.parameters(),
                    max_norm=FLAGS.grad_clip_norm
                )
                print('[debug] clip_grad_norm return:', float(clip_return))

                optim.step()
                batch_idx += 1

                pbar.set_postfix({
                    'loss': f'{total_loss.item():.3f}',
                    'spike': f'{actual_spike_rate:.2%}',
                    'lr': f'{optim.param_groups[0]["lr"]:.2e}',
                })

            except Exception:
                print(f'[debug] training crashed at batch_idx={batch_idx}')
                dump_batch(batch, prefix=f'train_batch_{batch_idx}_crash')
                raise

        train_loss = float(np.mean(losses)) if losses else float('nan')

        if FLAGS.debug_skip_validation:
            val = float('nan')
            phoneme_acc = float('nan')
            logging.info(
                f'finished epoch {epoch_idx + 1} - validation skipped - training loss: {train_loss:.4f}'
            )
        else:
            try:
                val, phoneme_acc, _ = test(model_student, devset, device)
                lr_sched.step(val)
                logging.info(
                    f'finished epoch {epoch_idx + 1} - validation loss: {val:.4f} '
                    f'training loss: {train_loss:.4f} phoneme accuracy: {phoneme_acc * 100:.2f}'
                )
            except Exception:
                logging.exception('[debug] validation crashed')
                raise

        torch.save(model_student.state_dict(), os.path.join(FLAGS.output_directory, 'model.pt'))

        if stop_training_early:
            break

        if save_sound_outputs and vocoder is not None:
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

    return model_student


def main():
    if getattr(FLAGS, 'debug_cuda_launch_blocking', False):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logging.warning('[debug] CUDA_LAUNCH_BLOCKING=1 enabled')

    os.makedirs(FLAGS.output_directory, exist_ok=True)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
        ],
        level=logging.INFO,
        format='%(message)s'
    )

    logging.info(
        subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE,
            universal_newlines=True
        ).stdout
    )
    logging.info(
        subprocess.run(
            ['git', 'diff'],
            stdout=subprocess.PIPE,
            universal_newlines=True
        ).stdout
    )
    logging.info(sys.argv)

    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d', len(trainset), len(devset))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_model(
        trainset,
        devset,
        device,
        save_sound_outputs=(getattr(FLAGS, 'hifigan_checkpoint', None) is not None)
    )


if __name__ == '__main__':
    FLAGS(sys.argv)
    main()