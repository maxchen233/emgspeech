import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Any, Dict, List


class ShapeHookCounter:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.handles = []

    def reset(self):
        self.records = []

    def _shape_str(self, x):
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, (list, tuple)):
            out = []
            for item in x:
                if isinstance(item, torch.Tensor):
                    out.append(tuple(item.shape))
            return out if out else str(type(x))
        return str(type(x))

    def _get_main_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for item in x:
                if isinstance(item, torch.Tensor):
                    return item
        return None

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            if not name:
                continue
            if isinstance(module, (nn.Conv1d, nn.BatchNorm1d, nn.Linear, nn.LayerNorm, nn.Dropout)) or module.__class__.__name__ in {
                'SILIFNeuron', 'Mamba', 'ResBlock', 'MambaBlock', 'PureMambaBlock'
            }:
                h = module.register_forward_hook(self._make_hook(name, module))
                self.handles.append(h)

    def _make_hook(self, name: str, module: nn.Module):
        def hook(mod, inp, out):
            x = self._get_main_tensor(inp)
            y = self._get_main_tensor(out)
            rec = OrderedDict()
            rec['name'] = name
            rec['type'] = module.__class__.__name__
            rec['input_shape'] = self._shape_str(inp)
            rec['output_shape'] = self._shape_str(out)
            rec['estimated_ops'] = self._estimate_ops(module, x, y)
            self.records.append(rec)
        return hook

    def _estimate_ops(self, module: nn.Module, x: torch.Tensor, y: torch.Tensor):
        if x is None or y is None:
            return None

        if isinstance(module, nn.Conv1d):
            b, cin, _ = x.shape
            _, cout, lout = y.shape
            k = module.kernel_size[0]
            groups = module.groups
            return int(b * lout * cout * (cin // groups) * k)

        if isinstance(module, nn.Linear):
            *prefix, din = x.shape
            dout = y.shape[-1]
            n = 1
            for v in prefix:
                n *= v
            return int(n * din * dout)

        if isinstance(module, nn.BatchNorm1d):
            return int(y.numel() * 4)

        if isinstance(module, nn.LayerNorm):
            return int(y.numel() * 7)

        if isinstance(module, nn.Dropout):
            return 0

        if module.__class__.__name__ == 'SILIFNeuron':
            return int(y.numel() * 3)

        if module.__class__.__name__ == 'Mamba':
            if y.dim() != 3:
                return None
            b, t, d = y.shape
            d_state = getattr(module, 'd_state', None)
            d_conv = getattr(module, 'd_conv', None)
            d_inner = getattr(module, 'd_inner', d)
            if d_state is None or d_conv is None:
                return int(b * t * d * d)
            proj_in = b * t * d * (2 * d_inner)
            conv = b * t * d_inner * d_conv
            ssm = b * t * d_inner * d_state * 3
            proj_out = b * t * d_inner * d
            return int(proj_in + conv + ssm + proj_out)

        return None

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def summary(self):
        total = 0
        by_type = OrderedDict()
        for r in self.records:
            v = r['estimated_ops'] if r['estimated_ops'] is not None else 0
            total += v
            by_type[r['type']] = by_type.get(r['type'], 0) + v
        return {
            'total_estimated_ops': total,
            'by_type': by_type,
            'num_records': len(self.records),
        }