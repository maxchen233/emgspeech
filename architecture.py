import json
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba

from absl import flags

FLAGS = flags.FLAGS

# �踹��� notebook / �滩� import ���銴��蝢� flag
if "model_size" not in FLAGS:
    flags.DEFINE_integer("model_size", 512, "number of hidden dimensions")
if "num_layers" not in FLAGS:
    flags.DEFINE_integer("num_layers", 4, "number of layers")
if "dropout" not in FLAGS:
    flags.DEFINE_float("dropout", 0.2, "dropout")


def _strip_module_prefix(state_dict):
    """蝘駁膄 DataParallel/DDP �Ｙ��� module. �滨韌��"""
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dict")

    out = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        out[key] = value
    return out


def _extract_state_dict(checkpoint):
    """
    �舀螱嚗�
      1. checkpoint �祈澈撠望糓 state_dict
      2. {"state_dict": ...}
      3. {"model_state_dict": ...}
      4. {"model": ...}
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}"
        )

    # 撣貉� wrapper
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and any(
            torch.is_tensor(v) for v in value.values()
        ):
            return _strip_module_prefix(value)

    # plain state_dict
    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return _strip_module_prefix(checkpoint)

    raise ValueError(
        "Cannot find state_dict in checkpoint. "
        "Expected a plain state_dict or state_dict/model_state_dict/model key."
    )


def infer_pruned_cnn_config(checkpoint_path):
    """
    �湔𦻖敺𧼮�雿萄��� .pt checkpoint �冽𪃾�芣� CNN 蝯鞉���

    Returns
    -------
    model_size : int
    mid_channels : list[int]
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)

    if "w_raw_in.weight" not in state:
        raise KeyError("checkpoint missing w_raw_in.weight")

    model_size = int(state["w_raw_in.weight"].shape[0])

    mid_channels = []
    for i in range(3):
        key = f"conv_blocks.{i}.conv1.weight"
        if key not in state:
            raise KeyError(f"checkpoint missing {key}")
        mid_channels.append(int(state[key].shape[0]))

    return model_size, mid_channels


def load_prune_config(config_path):
    """
    霈� merge_cnn_mamba.py �Ｙ��� *_prune_config.json��
    """
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "mid_channels" not in cfg:
        raise KeyError(
            f"{config_path} does not contain 'mid_channels'"
        )

    mid_channels = [int(x) for x in cfg["mid_channels"]]

    if len(mid_channels) != 3:
        raise ValueError(
            f"Expected 3 mid_channels, got {mid_channels}"
        )

    model_size = int(cfg.get("model_size", 512))
    return model_size, mid_channels


def infer_mid_channels_from_state_dict(state_dict):
    """Infer pruned CNN internal widths from a full model state_dict."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value

    mids = []
    for i in range(3):
        key = f"conv_blocks.{i}.conv1.weight"
        if key not in normalized:
            raise KeyError(f"Cannot infer pruned CNN structure: missing {key}")
        mids.append(int(normalized[key].shape[0]))
    return mids


class ResBlock(nn.Module):
    """
    CNN 畾睃榆��憛𨳍��

    �𦠜芋�页�
        num_ins -> num_outs -> num_outs

    Network Slimming 敺䕘�
        num_ins -> mid_channels -> num_outs

    瘜冽�嚗�
    num_outs 隞滨雁�� model_size (靘见� 512)嚗峕�隞� CNN 頛詨枂蝬剖漲
    ����Ｙ� w_raw_in / Mamba 摰���詨捆��
    """

    def __init__(
        self,
        num_ins,
        num_outs,
        stride=1,
        mid_channels=None,
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = num_outs

        self.num_ins = int(num_ins)
        self.num_outs = int(num_outs)
        self.mid_channels = int(mid_channels)
        self.stride = int(stride)

        # �芣���迤�潛��券�躰ㄐ嚗�
        # conv1 output channel / BN channel / conv2 input channel
        self.conv1 = nn.Conv1d(
            self.num_ins,
            self.mid_channels,
            kernel_size=3,
            padding=1,
            stride=self.stride,
        )
        self.bn1 = nn.BatchNorm1d(self.mid_channels)

        self.conv2 = nn.Conv1d(
            self.mid_channels,
            self.num_outs,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(self.num_outs)

        # residual branch 頛詨枂隞滨��� num_outs
        if self.stride != 1 or self.num_ins != self.num_outs:
            self.residual_path = nn.Conv1d(
                self.num_ins,
                self.num_outs,
                kernel_size=1,
                stride=self.stride,
            )
            self.res_norm = nn.BatchNorm1d(self.num_outs)
        else:
            self.residual_path = None
            self.res_norm = None

    def forward(self, x):
        """�寞活璅∪�嚗��蝺� / 銝��祆綫隢吔�"""
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

    def forward_causal(self, x, cache1, cache2):
        """
        �䭾�銝脫�璅∪�嚗�蘨雿輻鍂撌血�甇瑕蟮鞈����

        Parameters
        ----------
        x:
            (B, C_in, T)
        cache1:
            (B, C_in, 2)嚗稪onv1 ��風�� cache
        cache2:
            (B, C_mid, 2)嚗稪onv2 ��風�� cache

        Returns
        -------
        output:
            (B, C_out, T_out)
        new_cache1:
            conv1 cache
        new_cache2:
            conv2 cache
        """
        input_value = x
        k = 3

        # ������������������������������������������������������������������������������������������������������������
        # conv1 causal
        # input channels = num_ins
        # output channels = mid_channels
        # ������������������������������������������������������������������������������������������������������������
        if cache1 is not None:
            x1 = torch.cat([cache1, x], dim=-1)
        else:
            x1 = F.pad(x, (k - 1, 0))

        new_cache1 = x1[:, :, -(k - 1):].clone()

        x = F.conv1d(
            x1,
            self.conv1.weight,
            self.conv1.bias,
            stride=self.conv1.stride[0],
        )
        x = F.relu(self.bn1(x))

        # ������������������������������������������������������������������������������������������������������������
        # conv2 causal
        # input channels = mid_channels
        # output channels = num_outs
        # ������������������������������������������������������������������������������������������������������������
        if cache2 is not None:
            x2 = torch.cat([cache2, x], dim=-1)
        else:
            x2 = F.pad(x, (k - 1, 0))

        new_cache2 = x2[:, :, -(k - 1):].clone()

        x = F.conv1d(
            x2,
            self.conv2.weight,
            self.conv2.bias,
            stride=self.conv2.stride[0],
        )
        x = self.bn2(x)

        # residual branch
        if self.residual_path is not None:
            res = self.res_norm(
                self.residual_path(input_value)
            )
        else:
            res = input_value

        return F.relu(x + res), new_cache1, new_cache2


class MambaBlock(nn.Module):
    """Mamba SSM ��憛� - �舀螱�寞活��葡瘚��蝔格芋撘�"""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=8,
            d_conv=4,
            expand=1,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x

    def forward_streaming(
        self,
        x,
        conv_state,
        ssm_state,
    ):
        """
        銝脫�璅∪�嚗帋蝙�� Mamba.step()��
        """
        _, T, _ = x.shape

        residual = x
        x = self.norm(x)

        outputs = []

        for t in range(T):
            xt = x[:, t:t + 1, :]

            out_t, conv_state, ssm_state = self.mamba.step(
                xt,
                conv_state,
                ssm_state,
            )

            outputs.append(out_t.squeeze(1))

        x = torch.stack(outputs, dim=1)

        # streaming inference 銝滢蝙�� dropout
        return residual + x, conv_state, ssm_state


class Model(nn.Module):
    """
    CNN + Mamba 璅∪���

    �𦠜芋�讠㮾摰對�
        model = Model(num_features, num_outs, num_aux_outs)

    �芣�璅∪��寞� 1嚗�綫�佗��芣�銝��� .pt 撠勗�嚗㚁�
        model = Model(
            num_features,
            num_outs,
            num_aux_outs,
            checkpoint_path="model_cnn_pruned_mamba.pt",
        )

    �芣�璅∪��寞� 2嚗�蝙�� merge script �Ｙ��� JSON嚗㚁�
        model = Model(
            num_features,
            num_outs,
            num_aux_outs,
            prune_config="model_cnn_pruned_mamba_prune_config.json",
        )

    �芣�璅∪��寞� 3嚗���閙�摰𡄯�嚗�
        model = Model(
            num_features,
            num_outs,
            num_aux_outs,
            mid_channels=[m0, m1, m2],
        )
    """

    def __init__(
        self,
        num_features,
        num_outs,
        num_aux_outs=None,
        mid_channels=None,
        prune_config=None,
        checkpoint_path=None,
        model_size=None,
    ):
        super().__init__()

        # ------------------------------------------------------
        # 1. 瘙箏� CNN 蝯鞉�
        # ------------------------------------------------------
        inferred_model_size = None
        inferred_mid_channels = None

        if checkpoint_path is not None:
            (
                inferred_model_size,
                inferred_mid_channels,
            ) = infer_pruned_cnn_config(checkpoint_path)

        if prune_config is not None:
            (
                config_model_size,
                config_mid_channels,
            ) = load_prune_config(prune_config)

            if inferred_model_size is not None:
                if config_model_size != inferred_model_size:
                    raise ValueError(
                        "checkpoint_path and prune_config disagree on "
                        f"model_size: {inferred_model_size} vs "
                        f"{config_model_size}"
                    )

                if list(config_mid_channels) != list(
                    inferred_mid_channels
                ):
                    raise ValueError(
                        "checkpoint_path and prune_config disagree on "
                        f"mid_channels: {inferred_mid_channels} vs "
                        f"{config_mid_channels}"
                    )

            inferred_model_size = config_model_size
            inferred_mid_channels = config_mid_channels

        if model_size is None:
            if inferred_model_size is not None:
                model_size = inferred_model_size
            else:
                model_size = int(FLAGS.model_size)

        self.model_size = int(model_size)

        if mid_channels is None:
            if inferred_mid_channels is not None:
                mid_channels = inferred_mid_channels
            else:
                # 銝齿�摰� = �𦠜芋�讠�瑽页�摰���睲��詨捆
                mid_channels = [
                    self.model_size,
                    self.model_size,
                    self.model_size,
                ]

        self.mid_channels = [int(x) for x in mid_channels]

        if len(self.mid_channels) != 3:
            raise ValueError(
                "mid_channels must contain exactly 3 values, "
                f"got {self.mid_channels}"
            )

        if any(x <= 0 for x in self.mid_channels):
            raise ValueError(
                f"mid_channels must all be > 0: {self.mid_channels}"
            )

        # ------------------------------------------------------
        # 2. CNN frontend
        #
        # �滩�嚗�
        # 瘥誩�� block ����� channel �臭誑鋡怠��嘅�
        # 雿� block output 瘞賊�靽脲� model_size��
        # �䭾迨 Mamba 銝漤�閬�㺿��
        # ------------------------------------------------------
        self.conv_blocks = nn.Sequential(
            ResBlock(
                8,
                self.model_size,
                stride=2,
                mid_channels=self.mid_channels[0],
            ),
            ResBlock(
                self.model_size,
                self.model_size,
                stride=2,
                mid_channels=self.mid_channels[1],
            ),
            ResBlock(
                self.model_size,
                self.model_size,
                stride=2,
                mid_channels=self.mid_channels[2],
            ),
        )

        self.w_raw_in = nn.Linear(
            self.model_size,
            self.model_size,
        )

        # ------------------------------------------------------
        # 3. Mamba
        # ------------------------------------------------------
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=self.model_size,
                dropout=FLAGS.dropout,
            )
            for _ in range(FLAGS.num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.model_size)

        # ------------------------------------------------------
        # 4. output heads
        # ------------------------------------------------------
        self.w_out = nn.Linear(
            self.model_size,
            num_outs,
        )

        self.has_aux_out = num_aux_outs is not None

        if self.has_aux_out:
            self.w_aux = nn.Linear(
                self.model_size,
                num_aux_outs,
            )

    def print_structure(self):
        """�嫣噶蝣箄�撖阡�頛匧�����萘�瑽卝��"""
        print("=" * 60)
        print("CNN-Mamba architecture")
        print("=" * 60)
        print(f"model_size   : {self.model_size}")
        print(f"mid_channels : {self.mid_channels}")

        for i, block in enumerate(self.conv_blocks):
            print(
                f"Block {i}: "
                f"{block.num_ins} -> "
                f"{block.mid_channels} -> "
                f"{block.num_outs}, "
                f"stride={block.stride}"
            )

        print(f"Mamba layers : {len(self.mamba_layers)}")
        print("=" * 60)

    def _rebuild_cnn(self, mid_channels):
        """Rebuild only CNN blocks with pruned internal channels."""
        self.mid_channels = [int(x) for x in mid_channels]
        if len(self.mid_channels) != 3:
            raise ValueError(f"mid_channels must contain 3 values, got {self.mid_channels}")

        self.conv_blocks = nn.Sequential(
            ResBlock(
                8,
                self.model_size,
                stride=2,
                mid_channels=self.mid_channels[0],
            ),
            ResBlock(
                self.model_size,
                self.model_size,
                stride=2,
                mid_channels=self.mid_channels[1],
            ),
            ResBlock(
                self.model_size,
                self.model_size,
                stride=2,
                mid_channels=self.mid_channels[2],
            ),
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Automatically detect pruned CNN widths before loading weights.

        This keeps old inference code working:
            model = Model(...)
            model.load_state_dict(state_dict)
        """
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            normalized[key] = value

        key0 = "conv_blocks.0.conv1.weight"
        if key0 in normalized:
            checkpoint_mids = infer_mid_channels_from_state_dict(normalized)

            if checkpoint_mids != self.mid_channels:
                print(
                    f"[architecture] Rebuilding CNN: "
                    f"{self.mid_channels} -> {checkpoint_mids}"
                )

                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")

                self._rebuild_cnn(checkpoint_mids)
                self.conv_blocks.to(device)

        try:
            return super().load_state_dict(
                normalized,
                strict=strict,
                assign=assign,
            )
        except TypeError:
            return super().load_state_dict(
                normalized,
                strict=strict,
            )

    def load_checkpoint(
        self,
        checkpoint_path,
        map_location="cpu",
        strict=True,
    ):
        """
        頛匧��笔��� merge_cnn_mamba.py �Ｙ��� checkpoint��
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
        )

        state = _extract_state_dict(checkpoint)

        result = self.load_state_dict(
            state,
            strict=strict,
        )

        return result

    def forward(
        self,
        x_feat,
        x_raw,
        session_ids,
    ):
        """
        �寞活璅∪���

        x_feat / session_ids 靽萘�嚗�
        隞亦雁����暹� training / inference call signature��
        """
        if self.training:
            r = random.randrange(8)

            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]
                x_raw[:, -r:, :] = 0

        # (B, T, 8) -> (B, 8, T)
        x_raw = x_raw.transpose(1, 2)

        # CNN
        x_raw = self.conv_blocks(x_raw)

        # (B, 512, T') -> (B, T', 512)
        x_raw = x_raw.transpose(1, 2)

        x = self.w_raw_in(x_raw)

        # Mamba
        for layer in self.mamba_layers:
            x = layer(x)

        x = self.final_norm(x)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)

        return self.w_out(x)

    def init_streaming_state(self, device):
        """
        �嘥��� CNN cache + Mamba state��
        """
        state = {
            "cnn_caches": [
                (None, None)
                for _ in range(len(self.conv_blocks))
            ],
            "mamba_states": [],
        }

        for layer in self.mamba_layers:
            d_inner = layer.mamba.d_inner
            d_conv = layer.mamba.d_conv
            d_state = layer.mamba.d_state

            conv_state = torch.zeros(
                1,
                d_inner,
                d_conv,
                device=device,
            )

            ssm_state = torch.zeros(
                1,
                d_inner,
                d_state,
                device=device,
            )

            state["mamba_states"].append(
                (conv_state, ssm_state)
            )

        return state

    def forward_streaming(
        self,
        x_raw,
        state,
    ):
        """
        Stateful causal streaming inference.

        Parameters
        ----------
        x_raw:
            (1, chunk_len, 8)
        state:
            init_streaming_state() �Ｙ��� state

        Returns
        -------
        mel:
            (1, chunk_frames, num_outs)
        aux:
            (1, chunk_frames, num_aux_outs) or None
        state:
            �湔鰵敺� state
        """
        # (1, T, 8) -> (1, 8, T)
        x = x_raw.transpose(1, 2)

        # CNN
        for i, block in enumerate(self.conv_blocks):
            cache1, cache2 = state["cnn_caches"][i]

            x, cache1, cache2 = block.forward_causal(
                x,
                cache1,
                cache2,
            )

            state["cnn_caches"][i] = (
                cache1,
                cache2,
            )

        # (1, C, T') -> (1, T', model_size)
        x = x.transpose(1, 2)

        x = self.w_raw_in(x)

        # Mamba
        for i, layer in enumerate(self.mamba_layers):
            conv_s, ssm_s = state["mamba_states"][i]

            x, conv_s, ssm_s = layer.forward_streaming(
                x,
                conv_s,
                ssm_s,
            )

            state["mamba_states"][i] = (
                conv_s,
                ssm_s,
            )

        x = self.final_norm(x)

        mel = self.w_out(x)

        if self.has_aux_out:
            aux = self.w_aux(x)
        else:
            aux = None

        return mel, aux, state