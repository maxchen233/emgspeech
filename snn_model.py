"""Spiking Neural Network (SNN) model for EMG-to-speech transduction.

Training methods implemented
----------------------------
BPTT + surrogate gradients
    The Heaviside firing function is non-differentiable, so gradients are
    approximated by smooth surrogate functions during the backward pass.
    Two surrogates are provided:
        * Triangular (piecewise-linear, SpikeProp-style)
        * ATan       (arc-tangent, SpikingJelly-style)

STDP (Spike-Timing-Dependent Plasticity)
    A local, online weight-update rule based on coincident pre/post spiking.
    Implemented with exponential eligibility traces.  STDP can run alongside
    (or instead of) BPTT to give a biologically-plausible component to learning.

SALSA (Surrogate-grAdient Learning with Spike Alignment)
    Adds a spike-rate regularisation loss that penalises the deviation of the
    mean firing rate from a target rate.  This prevents neuron collapse (all
    silent) or saturation (always firing) and is inspired by the SALSA family
    of SNN training algorithms.

References
----------
* Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks"
  https://arxiv.org/abs/1901.09948
* Bi & Poo (1998) "Synaptic modifications in cultured hippocampal neurons"
* Fang et al. (2021) SpikingJelly framework
  https://github.com/fangwei123456/spikingjelly
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# 1. Surrogate gradient functions   (BPTT / straight-through style)
# ─────────────────────────────────────────────────────────────────────────────

class _TriangularSurrogate(torch.autograd.Function):
    """Heaviside forward, piecewise-linear surrogate backward.

    Surrogate: g'(u) = max(0, 1 - |u - threshold| * slope)
    """
    slope: float = 25.0

    @staticmethod
    def forward(ctx, u: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (u,) = ctx.saved_tensors
        thresh = ctx.threshold
        surrogate = torch.clamp(
            1.0 - _TriangularSurrogate.slope * (u - thresh).abs(), min=0.0
        )
        return grad_output * surrogate, None


class _ATanSurrogate(torch.autograd.Function):
    """Heaviside forward, arc-tangent surrogate backward (SpikingJelly style).

    Surrogate: g'(u) = alpha / (2 * (1 + (pi/2 * alpha * (u-thresh))^2))
    """
    alpha: float = 2.0

    @staticmethod
    def forward(ctx, u: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (u,) = ctx.saved_tensors
        thresh = ctx.threshold
        alpha = _ATanSurrogate.alpha
        surrogate = (alpha / 2.0) / (
            1.0 + (math.pi / 2.0 * alpha * (u - thresh)).pow(2)
        )
        return grad_output * surrogate, None


def _get_spike_fn(name: str):
    """Return the .apply callable for the requested surrogate."""
    if name == 'triangular':
        return _TriangularSurrogate.apply
    if name == 'atan':
        return _ATanSurrogate.apply
    raise ValueError(f"Unknown surrogate '{name}'. Choose 'triangular' or 'atan'.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIF (Leaky Integrate-and-Fire) neuron layer
# ─────────────────────────────────────────────────────────────────────────────

class LIFNeuron(nn.Module):
    """Vectorised LIF neuron layer for a sequence of time-steps.

    Dynamics (soft-reset / reset-by-subtraction):
        mem[t] = decay * mem[t-1] + input[t]
        spike[t] = Θ(mem[t] - threshold)    # Heaviside with surrogate grad
        mem[t] -= spike[t] * threshold       # soft reset

    Soft reset keeps a continuous membrane trace which improves gradient flow
    compared to hard reset.

    Args:
        threshold  : firing threshold V_th  (default 1.0)
        decay      : membrane potential decay per step (default 0.9)
                     Relates to membrane time constant as τ_m ≈ -1 / ln(decay)
        surrogate  : 'atan' (default) | 'triangular'
    """

    def __init__(
        self,
        threshold: float = 1.0,
        decay: float = 0.9,
        surrogate: str = 'atan',
    ):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self._spike_fn = _get_spike_fn(surrogate)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:   (T, batch, neurons) – pre-synaptic input current each time-step
            mem: (batch, neurons)    – initial membrane potential; zeros if None

        Returns:
            spikes : (T, batch, neurons)  ∈ {0, 1}  (float for grad purposes)
            mem    : (batch, neurons)     final membrane potential
        """
        T = x.shape[0]
        if mem is None:
            mem = torch.zeros_like(x[0])

        spikes_list: list[torch.Tensor] = []
        for t in range(T):
            mem = self.decay * mem + x[t]                   # integrate
            spike = self._spike_fn(mem, self.threshold)     # fire (surrogate)
            mem = mem - spike * self.threshold              # soft reset
            spikes_list.append(spike)

        return torch.stack(spikes_list, dim=0), mem


# ─────────────────────────────────────────────────────────────────────────────
# 3. STDP learning rule  (online, local, unsupervised)
# ─────────────────────────────────────────────────────────────────────────────

class STDPLearner:
    """Spike-Timing-Dependent Plasticity weight updater.

    Uses the classical exponential trace formulation:

        trace_pre[t]  = τ_pre⁻¹  · trace_pre[t-1]  + spike_pre[t]
        trace_post[t] = τ_post⁻¹ · trace_post[t-1] + spike_post[t]

    Weight update per time-step:
        ΔW +=  A+  · trace_pre  ⊗ spike_post    (LTP – potentiation)
        ΔW -=  A-  · spike_pre  ⊗ trace_post    (LTD – depression)

    Call reset() at the start of each new sequence.
    Call step() once per time-step inside the training loop.

    Args:
        weight    : nn.Parameter to be updated in-place (no-grad)
        lr        : STDP learning rate
        tau_pre   : pre-synaptic trace time constant  (time-steps)
        tau_post  : post-synaptic trace time constant (time-steps)
        a_plus    : LTP amplitude
        a_minus   : LTD amplitude
    """

    def __init__(
        self,
        weight: nn.Parameter,
        lr: float = 1e-3,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.0105,
    ):
        self.weight = weight
        self.lr = lr
        self._decay_pre = math.exp(-1.0 / tau_pre)
        self._decay_post = math.exp(-1.0 / tau_post)
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.trace_pre: torch.Tensor | None = None
        self.trace_post: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset eligibility traces at the start of a new sequence."""
        self.trace_pre = None
        self.trace_post = None

    @torch.no_grad()
    def step(self, spike_pre: torch.Tensor, spike_post: torch.Tensor) -> None:
        """Accumulate one time-step of STDP.

        Args:
            spike_pre  : (batch, n_pre)
            spike_post : (batch, n_post)
        """
        if self.trace_pre is None:
            self.trace_pre = torch.zeros_like(spike_pre)
        if self.trace_post is None:
            self.trace_post = torch.zeros_like(spike_post)

        self.trace_pre = self._decay_pre * self.trace_pre + spike_pre
        self.trace_post = self._decay_post * self.trace_post + spike_post

        # ΔW: sum over batch then divide by batch size; transposed to match (out, in) weight layout
        batch = spike_pre.shape[0]
        dw_ltp = self.a_plus * torch.einsum('bi,bj->ji', spike_pre, self.trace_post) / batch   # (n_post, n_pre)
        dw_ltd = self.a_minus * torch.einsum('bi,bj->ji', self.trace_pre, spike_post) / batch  # (n_post, n_pre)
        self.weight.data.add_(self.lr * (dw_ltp - dw_ltd))


# ─────────────────────────────────────────────────────────────────────────────
# 4. SNN encoder block
# ─────────────────────────────────────────────────────────────────────────────

class SNNBlock(nn.Module):
    """One spiking encoder block: Linear → LIF.

    For BPTT training the surrogate gradient flows through the LIF spike
    function automatically.  For STDP, attach an STDPLearner to
    ``self.stdp_learner``; it will be called once per time-step.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        threshold: float = 1.0,
        decay: float = 0.9,
        surrogate: str = 'atan',
    ):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out, bias=True)
        self.lif = LIFNeuron(threshold=threshold, decay=decay, surrogate=surrogate)
        self.stdp_learner: STDPLearner | None = None

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x   : (T, batch, d_in)  – spiking (or float) input at each time-step
            mem : (batch, d_out)    – initial membrane potential; zeros if None

        Returns:
            spikes : (T, batch, d_out)
            mem    : (batch, d_out)  final membrane potential
        """
        h = self.fc(x)                  # (T, batch, d_out)  – linear per step
        spikes, mem = self.lif(h, mem)

        if self.stdp_learner is not None and self.training:
            self.stdp_learner.reset()
            for t in range(x.shape[0]):
                self.stdp_learner.step(x[t], spikes[t])

        return spikes, mem


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full SNN encoder  (drop-in for EMGMambaModel / TransformerEncoder)
# ─────────────────────────────────────────────────────────────────────────────

class SNNEncoder(nn.Module):
    """Multi-layer SNN encoder for EMG sequences.

    Interface mirrors ``nn.TransformerEncoder``:
        input  : (time, batch, d_model)
        output : (time, batch, d_model)

    Training modes
    --------------
    BPTT (always active)
        Gradients flow back through time via the surrogate derivative of the
        LIF spike function.
    STDP (opt-in via ``use_stdp=True``)
        A local STDP weight update is applied after each forward pass.  STDP
        operates without gradient and complements BPTT.
    SALSA (opt-in via ``salsa_rate_reg > 0`` in the training loop)
        A spike-rate regularisation loss (computed outside this module by
        ``salsa_spike_rate_loss``) encourages homeostatic firing rates.

    Args:
        d_model        : feature dimension (same in and out)
        n_layers       : number of spiking blocks
        hidden_scale   : width of each hidden spiking layer relative to d_model
        threshold      : LIF firing threshold
        decay          : LIF membrane decay factor
        surrogate      : surrogate gradient type ('atan' | 'triangular')
        dropout        : dropout applied after the output projection
        use_stdp       : attach STDP learners to each block's weight matrix
        use_checkpoint : apply gradient checkpointing per block to save VRAM
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        hidden_scale: float = 2.0,
        threshold: float = 1.0,
        decay: float = 0.9,
        surrogate: str = 'atan',
        dropout: float = 0.2,
        use_stdp: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint

        d_hidden = int(d_model * hidden_scale)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            d_in = d_model if i == 0 else d_hidden
            self.blocks.append(
                SNNBlock(d_in, d_hidden, threshold=threshold, decay=decay, surrogate=surrogate)
            )
        self.proj_out = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.last_spikes: list[torch.Tensor] = []   # populated each forward pass

        if use_stdp:
            for block in self.blocks:
                block.stdp_learner = STDPLearner(block.fc.weight)

    # helper so checkpoint can wrap a block cleanly (no tuple return)
    def _block_spikes(self, block: SNNBlock, x: torch.Tensor) -> torch.Tensor:
        spikes, _ = block(x)
        return spikes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (time, batch, d_model) – real-valued or spiking input

        Returns:
            (time, batch, d_model)
        """
        self.last_spikes = []
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(self._block_spikes, block, x, use_reentrant=False)
            else:
                x, _ = block(x)
            self.last_spikes.append(x)

        # project spiking representations back to continuous feature space
        x = self.proj_out(x)    # (T, batch, d_model)
        x = self.dropout(x)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6. SALSA auxiliary loss
# ─────────────────────────────────────────────────────────────────────────────

def salsa_spike_rate_loss(spikes: torch.Tensor, target_rate: float = 0.1) -> torch.Tensor:
    """SALSA-style spike-rate regularisation.

    Penalises the squared deviation of the mean firing rate from a target.
    This prevents neuron collapse (silent) or saturation (always active) and
    is consistent with the homeostatic objectives in SALSA-style training.

    Args:
        spikes      : any tensor of spike values in {0, 1}
        target_rate : desired mean firing rate in [0, 1]  (default 10 %)

    Returns:
        scalar loss (to be added to the main task loss with a small weight)
    """
    mean_rate = spikes.float().mean()
    return (mean_rate - target_rate).pow(2)
