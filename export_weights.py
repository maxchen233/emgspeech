#!/usr/bin/env python3
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json, os, sys
CHECKPOINT = 'models_mamba1/transduction_model/model.pt'
OUTPUT_DIR = 'exported_weights'
D_MODEL, NUM_LAYERS, D_STATE, D_CONV = 512, 4, 8, 4
D_INNER, DT_RANK, NUM_OUTS, NUM_AUX = 512, 32, 80, 48
def fuse_bn(cw, cb, bw, bb, bm, bv, eps=1e-5):
    s = bw / torch.sqrt(bv + eps)
    return cw * s.view(-1,1,1), s * (cb - bm) + bb
def save_t(t, fp):
    a = t.detach().cpu().contiguous().float().numpy(); a.tofile(fp); return list(a.shape)
def mamba_forward_cpu(x, w):
    """Pure-Python Mamba forward (no CUDA). x: (B, L, D)"""
    B, L, D = x.shape
    residual = x
    # pre-norm
    x = F.layer_norm(x, [D], weight=w['norm_w'], bias=w['norm_b'])
    # in_proj: (B, L, D) -> (B, L, 2*D_INNER)
    xz = x @ w['in_proj_w'].T  # (B,L,2*d_inner)
    x_part, z = xz.chunk(2, dim=-1)  # each (B,L,d_inner)
    # conv1d (causal, groups=d_inner)
    x_conv = x_part.transpose(1, 2)  # (B, d_inner, L)
    x_conv = F.conv1d(x_conv, w['conv1d_w'], w['conv1d_b'], padding=D_CONV-1, groups=D_INNER)
    x_conv = x_conv[:, :, :L]  # causal: trim right
    x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)
    # x_proj -> delta, B_mat, C_mat
    x_dbc = x_conv @ w['x_proj_w'].T  # (B, L, dt_rank+2*d_state)
    delta_input = x_dbc[:, :, :DT_RANK]
    B_mat = x_dbc[:, :, DT_RANK:DT_RANK+D_STATE]
    C_mat = x_dbc[:, :, DT_RANK+D_STATE:]
    # dt_proj
    delta = delta_input @ w['dt_proj_w'].T + w['dt_proj_b']  # (B, L, d_inner)
    delta = F.softplus(delta)
    # A = -exp(A_log)
    A = -torch.exp(w['A_log'].float())  # (d_inner, d_state)
    D_param = w['D_param']
    # Selective scan (sequential)
    h = torch.zeros(B, D_INNER, D_STATE, dtype=x.dtype)  # SSM state
    ys = []
    for t in range(L):
        dt = delta[:, t, :]  # (B, d_inner)
        x_t = x_conv[:, t, :]  # (B, d_inner)
        B_t = B_mat[:, t, :]  # (B, d_state)
        C_t = C_mat[:, t, :]  # (B, d_state)
        # discretize
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)
        h = h * dA + dB * x_t.unsqueeze(-1)
        y_t = (h * C_t.unsqueeze(1)).sum(-1)  # (B, d_inner)
        y_t = y_t + D_param * x_t
        ys.append(y_t)
    y = torch.stack(ys, dim=1)  # (B, L, d_inner)
    # gate
    y = y * F.silu(z)
    # out_proj
    y = y @ w['out_proj_w'].T
    return residual + y
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sd = torch.load(CHECKPOINT, map_location='cpu')
    m = {'format': 'silentspeech_mamba_v1', 'd_model': D_MODEL, 'num_layers': NUM_LAYERS,
         'd_state': D_STATE, 'd_conv': D_CONV, 'd_inner': D_INNER, 'dt_rank': DT_RANK,
         'num_outs': NUM_OUTS, 'num_aux_outs': NUM_AUX, 'tensors': {}}
    def export(name, tensor):
        fn = name.replace('.', '_') + '.bin'
        fp = os.path.join(OUTPUT_DIR, fn)
        sh = save_t(tensor, fp)
        m['tensors'][name] = {'file': fn, 'shape': sh, 'numel': int(tensor.numel()), 'dtype': 'float32'}
        print(f'  {name:55s} {str(sh):20s}')
    print('=== ResBlocks (BN fused) ===')
    for i in range(3):
        p = f'conv_blocks.{i}'
        for tag, cv, bn in [('conv1','conv1','bn1'), ('conv2','conv2','bn2')]:
            fw, fb = fuse_bn(sd[f'{p}.{cv}.weight'], sd[f'{p}.{cv}.bias'], sd[f'{p}.{bn}.weight'], sd[f'{p}.{bn}.bias'], sd[f'{p}.{bn}.running_mean'], sd[f'{p}.{bn}.running_var'])
            export(f'res{i}.{tag}.weight', fw); export(f'res{i}.{tag}.bias', fb)
        if f'{p}.residual_path.weight' in sd:
            fw, fb = fuse_bn(sd[f'{p}.residual_path.weight'], sd[f'{p}.residual_path.bias'], sd[f'{p}.res_norm.weight'], sd[f'{p}.res_norm.bias'], sd[f'{p}.res_norm.running_mean'], sd[f'{p}.res_norm.running_var'])
            export(f'res{i}.skip.weight', fw); export(f'res{i}.skip.bias', fb)
            m['tensors'][f'res{i}.has_skip'] = True
        else:
            m['tensors'][f'res{i}.has_skip'] = False
        m['tensors'][f'res{i}.stride'] = 2
    print('=== w_raw_in ===')
    export('w_raw_in.weight', sd['w_raw_in.weight']); export('w_raw_in.bias', sd['w_raw_in.bias'])
    print('=== Mamba Layers ===')
    for i in range(NUM_LAYERS):
        p = f'mamba_layers.{i}'
        for suffix in ['norm.weight','norm.bias']:
            export(f'mamba{i}.{suffix}', sd[f'{p}.{suffix}'])
        export(f'mamba{i}.in_proj.weight', sd[f'{p}.mamba.in_proj.weight'])
        export(f'mamba{i}.conv1d.weight', sd[f'{p}.mamba.conv1d.weight'])
        export(f'mamba{i}.conv1d.bias', sd[f'{p}.mamba.conv1d.bias'])
        export(f'mamba{i}.x_proj.weight', sd[f'{p}.mamba.x_proj.weight'])
        export(f'mamba{i}.dt_proj.weight', sd[f'{p}.mamba.dt_proj.weight'])
        export(f'mamba{i}.dt_proj.bias', sd[f'{p}.mamba.dt_proj.bias'])
        export(f'mamba{i}.A_log', sd[f'{p}.mamba.A_log'])
        export(f'mamba{i}.D', sd[f'{p}.mamba.D'])
        export(f'mamba{i}.out_proj.weight', sd[f'{p}.mamba.out_proj.weight'])
    print('=== Output layers ===')
    export('final_norm.weight', sd['final_norm.weight']); export('final_norm.bias', sd['final_norm.bias'])
    export('w_out.weight', sd['w_out.weight']); export('w_out.bias', sd['w_out.bias'])
    export('w_aux.weight', sd['w_aux.weight']); export('w_aux.bias', sd['w_aux.bias'])
    # === Golden test with pure-Python Mamba ===
    print('=== Golden test (pure-Python Mamba, CPU) ===')
    torch.manual_seed(42); test_input = torch.randn(1, 128, 8)
    export('golden.input', test_input.squeeze(0))
    # ResBlock forward (BN fused version)
    def resblock_fwd(x_in, idx):
        p = f'conv_blocks.{idx}'
        fw1, fb1 = fuse_bn(sd[f'{p}.conv1.weight'], sd[f'{p}.conv1.bias'], sd[f'{p}.bn1.weight'], sd[f'{p}.bn1.bias'], sd[f'{p}.bn1.running_mean'], sd[f'{p}.bn1.running_var'])
        fw2, fb2 = fuse_bn(sd[f'{p}.conv2.weight'], sd[f'{p}.conv2.bias'], sd[f'{p}.bn2.weight'], sd[f'{p}.bn2.bias'], sd[f'{p}.bn2.running_mean'], sd[f'{p}.bn2.running_var'])
        inp = x_in
        x = F.relu(F.conv1d(x_in, fw1, fb1, stride=2, padding=1))
        x = F.conv1d(x, fw2, fb2, stride=1, padding=1)
        if f'{p}.residual_path.weight' in sd:
            fws, fbs = fuse_bn(sd[f'{p}.residual_path.weight'], sd[f'{p}.residual_path.bias'], sd[f'{p}.res_norm.weight'], sd[f'{p}.res_norm.bias'], sd[f'{p}.res_norm.running_mean'], sd[f'{p}.res_norm.running_var'])
            res = F.conv1d(inp, fws, fbs, stride=2)
        else:
            res = inp
        return F.relu(x + res)
    x = test_input.transpose(1, 2)  # (1, 8, 128)
    for bi in range(3):
        x = resblock_fwd(x, bi)
        export(f'golden.after_res{bi}', x.squeeze(0).transpose(0, 1))
    x = x.transpose(1, 2)  # (1, T, 512)
    x = x @ sd['w_raw_in.weight'].T + sd['w_raw_in.bias']
    export('golden.after_w_raw_in', x.squeeze(0))
    for li in range(NUM_LAYERS):
        p = f'mamba_layers.{li}'
        w = {
            'norm_w': sd[f'{p}.norm.weight'], 'norm_b': sd[f'{p}.norm.bias'],
            'in_proj_w': sd[f'{p}.mamba.in_proj.weight'],
            'conv1d_w': sd[f'{p}.mamba.conv1d.weight'], 'conv1d_b': sd[f'{p}.mamba.conv1d.bias'],
            'x_proj_w': sd[f'{p}.mamba.x_proj.weight'],
            'dt_proj_w': sd[f'{p}.mamba.dt_proj.weight'], 'dt_proj_b': sd[f'{p}.mamba.dt_proj.bias'],
            'A_log': sd[f'{p}.mamba.A_log'], 'D_param': sd[f'{p}.mamba.D'],
            'out_proj_w': sd[f'{p}.mamba.out_proj.weight'],
        }
        x = mamba_forward_cpu(x, w)
        export(f'golden.after_mamba{li}', x.squeeze(0))
    x = F.layer_norm(x, [D_MODEL], sd['final_norm.weight'], sd['final_norm.bias'])
    export('golden.after_final_norm', x.squeeze(0))
    mel = x @ sd['w_out.weight'].T + sd['w_out.bias']
    aux = x @ sd['w_aux.weight'].T + sd['w_aux.bias']
    export('golden.mel_output', mel.squeeze(0))
    export('golden.aux_output', aux.squeeze(0))

    # Golden Streaming
    from absl import flags
    try:
        flags.FLAGS(['export_weights.py'])
    except Exception:
        pass
    try:
        flags.FLAGS.model_size = D_MODEL
        flags.FLAGS.num_layers = NUM_LAYERS
        flags.FLAGS.dropout = 0.0
    except Exception:
        pass

    sys.path.insert(0, '.')
    from architecture import Model
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_st = Model(num_features=8, num_outs=NUM_OUTS, num_aux_outs=NUM_AUX).to(dev)
    model_st.load_state_dict(sd, strict=True)
    model_st.eval()
    st_input = test_input.to(dev)
    with torch.no_grad():
        st_state = model_st.init_streaming_state(dev)
        mel_st, _, _ = model_st.forward_streaming(st_input, st_state)
        export('golden.mel_stream_output', mel_st.squeeze(0).cpu())

    with open(os.path.join(OUTPUT_DIR, 'model.json'), 'w') as f: json.dump(m, f, indent=2)
    tb = sum(v['numel']*4 for v in m['tensors'].values() if isinstance(v, dict) and 'numel' in v)
    nf = sum(1 for v in m['tensors'].values() if isinstance(v, dict))
    print(f'\nExported {nf} tensors, {tb/1024/1024:.2f} MB total')
    print('Done.')
if __name__ == '__main__': main()