"""EMG-specific Mamba sequence model with VRAM optimisations.

Drop-in replacement for nn.TransformerEncoder in architecture.py.
Interface: forward(x) where x is (time, batch, d_model).

VRAM optimisations
------------------
* Gradient checkpointing per layer (set use_checkpoint=True in FLAGS or constructor).
* scan_mode='cumsum' avoids the large complex-number intermediates produced by
  the default logcumsumexp mode – saves memory at the cost of slight numerical
  precision on very long sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model import MambaBlock, ModelArgs, RMSNorm

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_bool('mamba_use_checkpoint', False,
                  'enable gradient checkpointing in EMGMambaModel to reduce VRAM')
flags.DEFINE_string('mamba_scan_mode', 'cumsum',
                    'selective-scan mode: "cumsum" (less VRAM) or "logcumsumexp" (more stable)')
flags.DEFINE_integer('mamba_d_state', 16, 'SSM latent state dimension')
flags.DEFINE_integer('mamba_expand', 2, 'Mamba expansion factor')
flags.DEFINE_integer('mamba_d_conv', 4, 'Mamba conv kernel size')


class EMGMambaLayer(nn.Module):
    """Pre-norm residual Mamba layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, d_model)
        return self.mixer(self.norm(x)) + x


class EMGMambaModel(nn.Module):
    """Lightweight Mamba-based sequence model for EMG signals.

    Accepts (time, batch, d_model) tensors – the same interface used by
    nn.TransformerEncoder – and returns a tensor of the same shape.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int | None = None,
        expand: int | None = None,
        d_conv: int | None = None,
        scan_mode: str | None = None,
        use_checkpoint: bool | None = None,
    ):
        super().__init__()

        # Prefer explicitly-passed values, fall back to FLAGS (registered above).
        _d_state = d_state if d_state is not None else FLAGS.mamba_d_state
        _expand = expand if expand is not None else FLAGS.mamba_expand
        _d_conv = d_conv if d_conv is not None else FLAGS.mamba_d_conv
        _scan_mode = scan_mode if scan_mode is not None else FLAGS.mamba_scan_mode
        _use_ckpt = use_checkpoint if use_checkpoint is not None else FLAGS.mamba_use_checkpoint

        args = ModelArgs(
            d_model=d_model,
            n_layer=n_layers,
            vocab_size=8,           # placeholder – not used in encoder mode
            d_state=_d_state,
            expand=_expand,
            d_conv=_d_conv,
            scan_mode=_scan_mode,
        )

        self.layers = nn.ModuleList([EMGMambaLayer(args) for _ in range(n_layers)])
        self.use_checkpoint = _use_ckpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (time, batch, d_model)

        Returns:
            (time, batch, d_model)
        """
        x = x.transpose(0, 1)          # → (batch, time, d_model)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return x.transpose(0, 1)       # → (time, batch, d_model)
