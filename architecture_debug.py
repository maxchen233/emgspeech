import random
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 512, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 4, 'number of layers')
flags.DEFINE_float('dropout', .2, 'dropout')


def _debug_tensor(tag, x):
    if x is None:
        print(f'[debug] {tag}: None')
        return
    finite = torch.isfinite(x).all().item()
    msg = f'[debug] {tag}: shape={tuple(x.shape)} device={x.device} training='
    if hasattr(x, 'requires_grad'):
        pass
    print(msg + ('?') + f' finite={finite}')
    if x.numel() > 0:
        xd = x.detach()
        if finite:
            print(f'[debug] {tag} stats: min={float(xd.min())} max={float(xd.max())} mean={float(xd.mean())}')
        else:
            print(f'[debug] {tag} has non-finite values')
            raise RuntimeError(f'Non-finite tensor at {tag}')


class SurrogateSymmetricSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, D=4):
        ctx.save_for_backward(x)
        ctx.D = D
        return torch.clamp(torch.round(x), -D, D)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.abs() > ctx.D] = 0
        return grad_input, None


class SILIFNeuron(nn.Module):
    def __init__(self, D=4):
        super().__init__()
        self.D = D

    def forward(self, x):
        return SurrogateSymmetricSpike.apply(x, self.D)


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None
            self.res_norm = None

    def forward(self, x):
        print('[debug] entered ResBlock.forward')
        print('[debug] ResBlock input:', x.shape, 'device', x.device, 'training', self.training,
              'finite', torch.isfinite(x).all().item())
        input_value = x

        x = self.conv1(x)
        print('[debug] conv1 out shape', x.shape, 'device', x.device, 'training', self.training,
              'finite', torch.isfinite(x).all().item())
        if not torch.isfinite(x).all():
            raise RuntimeError('NaN/Inf after conv1')

        print('[debug] bn1 running_mean finite?', torch.isfinite(self.bn1.running_mean).all())
        print('[debug] bn1 running_var  finite?', torch.isfinite(self.bn1.running_var).all())
        print('[debug] bn1 weight finite?', torch.isfinite(self.bn1.weight).all())
        print('[debug] bn1 bias   finite?', torch.isfinite(self.bn1.bias).all())
        x = self.bn1(x)
        print('[debug] after bn1 finite:', torch.isfinite(x).all().item())
        if not torch.isfinite(x).all():
            raise RuntimeError('NaN/Inf after bn1')

        x = F.relu(x)
        print('[debug] after relu1 finite:', torch.isfinite(x).all().item())

        x = self.conv2(x)
        print('[debug] conv2 out shape', x.shape, 'device', x.device, 'training', self.training,
              'finite', torch.isfinite(x).all().item())
        if not torch.isfinite(x).all():
            raise RuntimeError('NaN/Inf after conv2')

        print('[debug] bn2 running_mean finite?', torch.isfinite(self.bn2.running_mean).all())
        print('[debug] bn2 running_var  finite?', torch.isfinite(self.bn2.running_var).all())
        print('[debug] bn2 weight finite?', torch.isfinite(self.bn2.weight).all())
        print('[debug] bn2 bias   finite?', torch.isfinite(self.bn2.bias).all())
        x = self.bn2(x)
        print('[debug] after bn2 finite:', torch.isfinite(x).all().item())
        if not torch.isfinite(x).all():
            raise RuntimeError('NaN/Inf after bn2')

        if self.residual_path is not None:
            rp = self.residual_path(input_value)
            print('[debug] residual_path out shape:', rp.shape, 'device:', rp.device,
                  'training:', self.training, 'finite:', torch.isfinite(rp).all().item())
            if not torch.isfinite(rp).all():
                raise RuntimeError('NaN/Inf after residual_path')

            print('[debug] res_norm running_mean finite:', torch.isfinite(self.res_norm.running_mean).all().item())
            print('[debug] res_norm running_var finite:', torch.isfinite(self.res_norm.running_var).all().item())
            print('[debug] res_norm weight finite:', torch.isfinite(self.res_norm.weight).all().item())
            print('[debug] res_norm bias finite:', torch.isfinite(self.res_norm.bias).all().item())
            res = self.res_norm(rp)
            print('[debug] residual after norm finite:', torch.isfinite(res).all().item())
            if not torch.isfinite(res).all():
                raise RuntimeError('NaN/Inf after res_norm')
        else:
            res = input_value

        summed = x + res
        print('[debug] x + res finite:', torch.isfinite(summed).all().item())
        if not torch.isfinite(summed).all():
            raise RuntimeError('NaN/Inf after residual add')

        out = F.relu(summed)
        print('[debug] ResBlock output:', out.shape, 'device', out.device, 'training', self.training,
              'finite', torch.isfinite(out).all().item())
        return out

    def forward_causal(self, x, cache1, cache2):
        input_value = x
        k = 3

        if cache1 is not None:
            x1 = torch.cat([cache1, x], dim=-1)
        else:
            x1 = F.pad(x, (k - 1, 0))
        new_cache1 = x1[:, :, -(k - 1):].clone()

        x = F.conv1d(x1, self.conv1.weight, self.conv1.bias, stride=self.conv1.stride[0])
        x = F.relu(self.bn1(x))

        if cache2 is not None:
            x2 = torch.cat([cache2, x], dim=-1)
        else:
            x2 = F.pad(x, (k - 1, 0))
        new_cache2 = x2[:, :, -(k - 1):].clone()

        x = F.conv1d(x2, self.conv2.weight, self.conv2.bias, stride=self.conv2.stride[0])
        x = self.bn2(x)

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res), new_cache1, new_cache2


class PureMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)[0]
        return residual + self.dropout(x)


class PureMambaModel(nn.Module):
    def __init__(self, num_features, num_speech_features, num_phonemes):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)
        self.mamba_layers = nn.ModuleList([
            PureMambaBlock(d_model=FLAGS.model_size, dropout=FLAGS.dropout)
            for _ in range(FLAGS.num_layers)
        ])
        self.final_norm = nn.LayerNorm(FLAGS.model_size)
        self.w_out = nn.Linear(FLAGS.model_size, num_speech_features)
        self.w_aux = nn.Linear(FLAGS.model_size, num_phonemes)

    def forward(self, _, x_raw, session_ids):
        x = x_raw.transpose(1, 2)
        x = self.conv_blocks(x).transpose(1, 2)
        x = self.w_raw_in(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.w_out(x), self.w_aux(x)


class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, no_sgc=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.si_lif = SILIFNeuron(D=4)
        self.mamba = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)

        self.no_sgc = no_sgc
        if not self.no_sgc:
            self.mamba_sgc = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)

        self.dropout = nn.Dropout(dropout)
        self.current_hidden_loss = torch.tensor(0.0)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        if self.training:
            x_spike = self.si_lif(x)
            out_spike = self.mamba(x_spike)

            if self.no_sgc:
                self.current_hidden_loss = torch.tensor(0.0, device=x.device)
                x = out_spike
            else:
                x_sgc = 4.0 * torch.tanh(x)
                out_sgc = self.mamba_sgc(x_sgc)
                self.current_hidden_loss = F.mse_loss(out_spike, out_sgc)
                x = out_spike
        else:
            x = self.mamba(self.si_lif(x))

        return residual + self.dropout(x)

    def forward_streaming(self, x, conv_state, ssm_state):
        residual = x
        x = self.norm(x)
        outputs = []
        for t in range(x.shape[1]):
            xt = x[:, t:t + 1, :]
            xt_spike = self.si_lif(xt)
            out_t, conv_state, ssm_state = self.mamba.step(xt_spike, conv_state, ssm_state)
            outputs.append(out_t.squeeze(1))
        x = torch.stack(outputs, dim=1)
        return residual + x, conv_state, ssm_state


class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None, no_sgc=False):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.si_lif_in = SILIFNeuron(D=4)
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)
        self.use_sgc = not no_sgc
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=FLAGS.model_size,
                dropout=FLAGS.dropout,
                no_sgc=no_sgc,
            )
            for _ in range(FLAGS.num_layers)
        ])

        self.final_norm = nn.LayerNorm(FLAGS.model_size)
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)
        self.last_spike_tensor = None

    def forward(self, _, x_raw, session_ids):
        print('[debug] entered forward')

        if session_ids is not None:
            print('[debug] session_ids:', session_ids.shape, session_ids.device, 'training', self.training)
            if not torch.isfinite(session_ids).all():
                raise RuntimeError('Non-finite session_ids')

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r] = x_raw[:, r:]
                x_raw[:, -r:] = 0

        print('[debug] input x_raw:', x_raw.shape, x_raw.device, 'training', self.training,
              'finite', torch.isfinite(x_raw).all().item(),
              'min', float(x_raw.detach().min()), 'max', float(x_raw.detach().max()))

        x_raw = x_raw.transpose(1, 2)
        print('[debug] after transpose:', x_raw.shape, x_raw.device, 'finite', torch.isfinite(x_raw).all().item())

        for i, block in enumerate(self.conv_blocks):
            print(f'[debug] before conv block {i}:', x_raw.shape, x_raw.device,
                  'training', self.training, 'finite', torch.isfinite(x_raw).all().item())
            x_raw = block(x_raw)
            print(f'[debug] after conv block {i}:', x_raw.shape, x_raw.device,
                  'training', self.training, 'finite', torch.isfinite(x_raw).all().item(),
                  'min', float(x_raw.detach().min()), 'max', float(x_raw.detach().max()))
            if not torch.isfinite(x_raw).all():
                raise RuntimeError(f'Non-finite after conv block {i}')

        x_raw = x_raw.transpose(1, 2)
        print('[debug] after conv_blocks transpose back:', x_raw.shape, x_raw.device,
              'finite', torch.isfinite(x_raw).all().item())

        x_raw_spikes = self.si_lif_in(x_raw)
        self.last_spike_tensor = x_raw_spikes
        print('[debug] after si_lif_in:', x_raw_spikes.shape, x_raw_spikes.device,
              'finite', torch.isfinite(x_raw_spikes).all().item(), 'training', self.training)
        if not torch.isfinite(x_raw_spikes).all():
            raise RuntimeError('Non-finite after si_lif_in')

        x = self.w_raw_in(x_raw_spikes)
        print('[debug] after w_raw_in:', x.shape, x.device,
              'finite', torch.isfinite(x).all().item(), 'training', self.training)
        if not torch.isfinite(x).all():
            raise RuntimeError('Non-finite after w_raw_in')

        for i, layer in enumerate(self.mamba_layers):
            print(f'[debug] entering mamba layer {i}:', x.shape, x.device, self.training,
                  torch.isfinite(x).all().item())
            x = layer(x)
            print(f'[debug] after mamba layer {i}:', x.shape, x.device, torch.isfinite(x).all().item())
            if not torch.isfinite(x).all():
                raise RuntimeError(f'Non-finite after mamba layer {i}')

        x = self.final_norm(x)
        print('[debug] after final_norm:', x.shape, x.device, torch.isfinite(x).all().item())
        if not torch.isfinite(x).all():
            raise RuntimeError('Non-finite after final_norm')

        out = self.w_out(x)
        print('[debug] after w_out:', out.shape, out.device, torch.isfinite(out).all().item())
        if not torch.isfinite(out).all():
            raise RuntimeError('Non-finite after w_out')

        aux = self.w_aux(x) if self.has_aux_out else None
        if aux is not None:
            print('[debug] after w_aux:', aux.shape, aux.device, torch.isfinite(aux).all().item())
            if not torch.isfinite(aux).all():
                raise RuntimeError('Non-finite after w_aux')

        return out, aux