import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeLinearAC(nn.Module):
    def __init__(self, in_features, out_features, bias=True, spike_driven=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spike_driven = spike_driven
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, spike_driven={self.spike_driven}'

    def forward_dense(self, x):
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def forward_spike_driven(self, x):
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f'SpikeLinearAC expects [B,T,C] or [N,C], got {original_shape}')
        B, T, C = x.shape
        if C != self.in_features:
            raise ValueError(f'Input last dim {C} != in_features {self.in_features}')
        out = torch.zeros(B, T, self.out_features, device=x.device, dtype=self.weight.dtype)
        flat_x = x.reshape(-1, C)
        flat_out = out.reshape(-1, self.out_features)
        total_nonzero = 0
        max_nonzero = 0
        weight = self.weight.to(x.device)
        bias = self.bias.to(x.device) if self.bias is not None else None
        for n in range(flat_x.size(0)):
            s = flat_x[n]
            idx = torch.nonzero(s != 0, as_tuple=False).squeeze(-1)
            k = int(idx.numel())
            total_nonzero += k
            max_nonzero = max(max_nonzero, k)
            if k == 0:
                if bias is not None:
                    flat_out[n].copy_(bias)
                continue
            vals = s[idx].to(weight.dtype)
            cols = weight[:, idx]
            acc = (cols * vals.unsqueeze(0)).sum(dim=1)
            if bias is not None:
                acc = acc + bias
            flat_out[n].copy_(acc)
        self.last_total_nonzero = total_nonzero
        self.last_max_nonzero = max_nonzero
        self.last_tokens = flat_x.size(0)
        self.last_dense_mac = flat_x.size(0) * self.in_features * self.out_features
        self.last_sparse_ac = total_nonzero * self.out_features
        if len(original_shape) == 2:
            out = out.squeeze(1)
        return out

    def forward(self, x):
        if self.training or (not self.spike_driven):
            return self.forward_dense(x)
        return self.forward_spike_driven(x)


def replace_linear_with_spike_linear(module, target_names=None, prefix=''):
    for name, child in list(module.named_children()):
        full_name = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.Linear) and (target_names is None or name in target_names or full_name in target_names):
            new_layer = SpikeLinearAC(child.in_features, child.out_features, bias=(child.bias is not None), spike_driven=True)
            with torch.no_grad():
                new_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_spike_linear(child, target_names=target_names, prefix=full_name)


@torch.no_grad()
def collect_spike_linear_stats(module):
    rows = []
    for name, child in module.named_modules():
        if isinstance(child, SpikeLinearAC):
            rows.append({
                'name': name,
                'tokens': getattr(child, 'last_tokens', None),
                'total_nonzero': getattr(child, 'last_total_nonzero', None),
                'max_nonzero': getattr(child, 'last_max_nonzero', None),
                'dense_mac': getattr(child, 'last_dense_mac', None),
                'sparse_ac': getattr(child, 'last_sparse_ac', None),
            })
    return rows