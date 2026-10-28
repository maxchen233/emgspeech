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

class SurrogateSymmetricSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, D=4):
        ctx.save_for_backward(x)
        ctx.D = D
        # 隢𡝗���𡁏�𤏪�𡁜�𥟇崕鈭𥪜�交㟲�彍��碶蒂�𡁜�滨迂撘讛��� [-D, D]嚗䔶�萘�躰�䂿儔璆菜��
        return torch.clamp(torch.round(x), -D, D)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # ��瑼餌恣��嚗𡁶畆敶Ｖ誨��璇臬漲 (Surrogate Gradient)
        # 頞��枂 [-D, D] 蝭���滚��０摨行飛�妟嚗屸俈甇Ｘ０摨衣���
        grad_input[x.abs() > ctx.D] = 0
        return grad_input, None

class SILIFNeuron(nn.Module):
    def __init__(self, D=4):
        super().__init__()
        self.D = D

    def forward(self, x):
        return SurrogateSymmetricSpike.apply(x, self.D)

class ResBlock(nn.Module):
    """CNN 畾睃榆��憛� - �𣈲�螱�鸌甈∟��葡瘚��拍車璅∪��"""

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

    def forward(self, x):
        """�鸌甈⊥芋撘� (閮梶毀�鍂) - 撠滨迂 padding"""
        input_value = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value
        return F.relu(x + res)

    def forward_causal(self, x, cache1, cache2):
        """��䭾�靝葡瘚�璅∪�� - 撌血‵��, 銝滨�𧢲𧊋靘�"""
        input_value = x
        k = 3  # kernel_size

        # ���� conv1 (causal) ������������������������������������������������������������������������
        if cache1 is not None:
            x1 = torch.cat([cache1, x], dim=-1)
        else:
            x1 = F.pad(x, (k - 1, 0))
        new_cache1 = x1[:, :, -(k - 1):].clone()
        x = F.conv1d(x1, self.conv1.weight, self.conv1.bias,
                     stride=self.conv1.stride[0])
        x = F.relu(self.bn1(x))

        # ���� conv2 (causal) ������������������������������������������������������������������������
        if cache2 is not None:
            x2 = torch.cat([cache2, x], dim=-1)
        else:
            x2 = F.pad(x, (k - 1, 0))
        new_cache2 = x2[:, :, -(k - 1):].clone()
        x = F.conv1d(x2, self.conv2.weight, self.conv2.bias,
                     stride=self.conv2.stride[0])
        x = self.bn2(x)

        # ���� residual ����������������������������������������������������������������������������������
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res), new_cache1, new_cache2

class PureMambaBlock(nn.Module):
    """��撣怠��鍂��蝝� Mamba ��憛𠺪�峕�𣂷�𥡝��飛��煺�璅∩�璅��霈𦠜彍�𦶢��滨征��� (norm ��� mamba)"""
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # 蝝娪���� Mamba嚗����其�滚��鉄隞颱�� SI-LIF ���萘�䂿�枏����� SGC ���𣈲
        self.mamba = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x) # 100% 蝝娪���峕筑暺墧彍�讠��
        return residual + self.dropout(x)

class PureMambaModel(nn.Module):
    """璅蹱�㚚���� Mamba 璅∪�页�����函�∟���腈���� SGC嚗㚁������鍂雿𡏭䌊�𡢄擗曄� Teacher"""
    def __init__(self, num_features, num_speech_features, num_phonemes):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        # �鐤�㙈蝝娪���𣬚���憛𠺪�屸�蹱見��滨迂撠梯�質���𢠃�齿�𥪜�𣬚�𦒘�撠滢��龪��
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
        return (self.w_out(x), self.w_aux(x))

class MambaBlock(nn.Module):
    """Mamba SSM ��憛� - �𣈲�螱�鸌甈∟��葡瘚��拍車璅∪��"""
    def __init__(self, d_model, dropout=0.2, use_sgc=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
        # ���� 1. ���苷蜓頝臬�� ����
        self.si_lif = SILIFNeuron(D=4) # 隢𡝗��綫�㵽 D=4 
        self.mamba = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)
        
        self.use_sgc = use_sgc
        if self.use_sgc:
            self.mamba_sgc = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=1)
        else:
            self.mamba_sgc = None
        
        self.dropout = nn.Dropout(dropout)
        self.current_hidden_loss = torch.tensor(0.0) # ��嘥�见�㚚黸��誩惜 Loss

    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        if self.training:
            # �鞱�梶毀��麄�煾�躰�䔶蒂銵�
            # 銝餉楝敺𡢅�����嗪���𤏪��
            x_spike = self.si_lif(x)
            out_spike = self.mamba(x_spike)
            
            if self.use_sgc:
                x_sgc = 4.0 * torch.tanh(x)
                out_sgc = self.mamba_sgc(x_sgc)
                self.current_hidden_loss = F.mse_loss(out_spike, out_sgc)
            else:
                self.current_hidden_loss = out_spike.new_zeros(())
            
            x = out_spike
        else:
            # �鞉綫����麄�穃���其��� SGC ���𣈲嚗𣬚�磰���苷�衤辣撽���𤏪�𣬚���憭折�誯�讠��
            x = self.mamba(self.si_lif(x))
            self.current_hidden_loss = x.new_zeros(())
            
        return residual + self.dropout(x)

    def forward_streaming(self, x, conv_state, ssm_state):
        """銝脫�璅∪�� - �鍂 step() �鞉�甇亥�閧�, 蝬剛風 SSM ����"""
        residual = x
        x = self.norm(x)

        outputs = []
        for t in range(x.shape[1]):
            xt = x[:, t:t+1, :]  # (B, 1, D)
            
            # ���� 銝脫�蝡航���嘥�� ����
            # �銁�綫��銝脫�璅∪�譍�页���峕見敹�����㯄�舘���嘥�吔�䔶誑蝣箔�萘畆�腼�讠�𡑒�厩�箔�𤾸�蠘�� AC
            xt_spike = self.si_lif(xt)
            
            out_t, conv_state, ssm_state = self.mamba.step(
                xt_spike, conv_state, ssm_state
            )
            outputs.append(out_t.squeeze(1))  # (B, D)

        x = torch.stack(outputs, dim=1)  # (B, T, D)
        return residual + x, conv_state, ssm_state

class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None, no_sgc=False):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        
        # 鋆靝�羓�𡁏�批惜 upstream ��蟡䂿�枏��摰��𠺪�屸�踹�� AttributeError 
        self.si_lif_in = SILIFNeuron(D=4)
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        self.use_sgc = not no_sgc

        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=FLAGS.model_size,
                dropout=FLAGS.dropout,
                use_sgc=self.use_sgc
            )
            for _ in range(FLAGS.num_layers)
        ])

        self.final_norm = nn.LayerNorm(FLAGS.model_size)
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

        # �鍂�䲰餈質馱�瓲敹��䔄�怎��� tensor
        self.last_spike_tensor = None

    def forward(self, _, x_raw, session_ids):
        """�鸌甈⊥綫隢� (閮梶毀�鍂)"""
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:]
                x_raw[:,-r:,:] = 0

        # CNN ��滨垢��閧� (��𣂼�㚚�㗛�𡒊鸌敺�)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.conv_blocks(x_raw).transpose(1,2)
        
        # ���� ���肽�㗇�� ����
        x_raw_spikes = self.si_lif_in(x_raw) 
        self.last_spike_tensor = x_raw_spikes 

        # �凒�𦻖���亦�𡁏�扳�惩�撅歹�䔶�齿𣄽�𦻖
        x = self.w_raw_in(x_raw_spikes)

        # ���� MambaBlock �����
        for layer in self.mamba_layers:
            x = layer(x)

        x = self.final_norm(x)
        return (self.w_out(x), self.w_aux(x))

    def init_streaming_state(self, device):
        """��嘥�见�碶葡瘚��綫隢𣇉�����厩���"""
        state = {
            'cnn_caches': [(None, None) for _ in range(3)],
            'mamba_states': [],
        }

        for layer in self.mamba_layers:
            d_inner = layer.mamba.d_inner
            d_conv = layer.mamba.d_conv
            d_state = layer.mamba.d_state
            conv_state = torch.zeros(1, d_inner, d_conv, device=device)
            ssm_state = torch.zeros(1, d_inner, d_state, device=device)
            state['mamba_states'].append((conv_state, ssm_state))

        return state

    def forward_streaming(self, x_raw, state):
        """��䭾�靝葡瘚��綫隢�"""
        x = x_raw.transpose(1, 2)  # (1, 8, chunk_len)

        # ���� ��䭾�� CNN ��滨垢 ��������������������������������������������������������������������������
        for i, block in enumerate(self.conv_blocks):
            cache1, cache2 = state['cnn_caches'][i]
            x, cache1, cache2 = block.forward_causal(x, cache1, cache2)
            state['cnn_caches'][i] = (cache1, cache2)

        x = x.transpose(1, 2)  # (1, chunk_frames, 512)
        
        # ���� 銝脫�蝺𡁏�批惜 upstream ���嘥�� ����
        x_spikes = self.si_lif_in(x)
        x = self.w_raw_in(x_spikes)

        # ���� Stateful Mamba (�折�典歇�㟲���葡瘚����嘥��) ������������������������������������
        for i, layer in enumerate(self.mamba_layers):
            conv_s, ssm_s = state['mamba_states'][i]
            x, conv_s, ssm_s = layer.forward_streaming(x, conv_s, ssm_s)
            state['mamba_states'][i] = (conv_s, ssm_s)

        x = x.final_norm(x) if hasattr(x, 'final_norm') else self.final_norm(x)

        mel = self.w_out(x)
        aux = self.w_aux(x) if self.has_aux_out else None

        return mel, aux, state