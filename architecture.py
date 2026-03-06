import random

from torch import nn
import torch.nn.functional as F

from transformer import TransformerEncoderLayer
from emg_mamba import EMGMambaModel

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 256, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 5, 'number of layers')
flags.DEFINE_float('dropout', .2, 'dropout')
# New option to swap the sequence model
flags.DEFINE_bool('use_mamba', True, 'if True the transformer encoder is replaced with a lightweight MambaModel')

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

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        if FLAGS.use_mamba:
            # replace the transformer stack with our MambaModel
            self.sequence_model = EMGMambaModel(FLAGS.model_size, FLAGS.num_layers)
        else:
            encoder_layer = TransformerEncoderLayer(d_model=FLAGS.model_size, nhead=8, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=FLAGS.dropout)
            self.sequence_model = nn.TransformerEncoder(encoder_layer, FLAGS.num_layers)

        self.w_out = nn.Linear(FLAGS.model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0

        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw

        x = x.transpose(0,1) # put time first
        # whether transformer or mamba, the module expects (time,batch,dim)
        x = self.sequence_model(x)
        x = x.transpose(0,1)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        else:
            return self.w_out(x)

