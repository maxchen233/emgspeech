#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys

# 預設路徑設定
QAT_CHECKPOINT = 'output_qat_int8/qat_model_int16.pt'
OUTPUT_DIR = 'exported_weights_qat'
D_MODEL, NUM_LAYERS, D_STATE, D_CONV = 512, 4, 8, 4  # 配合 QAT 實際使用的 model_size (512)
D_INNER, DT_RANK, NUM_OUTS, NUM_AUX = 512, 32, 80, 48

def fuse_bn(cw, cb, bw, bb, bm, bv, eps=1e-5):
    s = bw / torch.sqrt(bv + eps)
    return cw * s.view(-1, 1, 1), s * (cb - bm) + bb

def quantize_tensor(param, bits=8):
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    # Per-channel 量化 (沿 dim=0)
    flat = param.view(param.size(0), -1)
    x_min = flat.min(dim=1).values
    x_max = flat.max(dim=1).values
    scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-12)
    zero_point = torch.round(qmin - x_min / scale)
    
    shape = [param.size(0)] + [1] * (param.dim() - 1)
    s = scale.view(shape)
    zp = zero_point.view(shape)
    
    q = torch.clamp(torch.round(param / s + zp), qmin, qmax).to(torch.int8)
    return q, scale, zero_point

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
    
    if not os.path.exists(QAT_CHECKPOINT):
        print(f"ERROR: Cannot find QAT checkpoint: {QAT_CHECKPOINT}")
        sys.exit(1)
        
    print(f"Loading QAT checkpoint from {QAT_CHECKPOINT}...")
    qat_data = torch.load(QAT_CHECKPOINT, map_location='cpu')
    
    q_weights = qat_data['quantized_weights']
    scales = qat_data['scales']
    other = qat_data['other_params']
    bits = qat_data.get('bits', 8)
    
    # 1. 重構完整的 FP32 state_dict
    sd = {}
    for name, q_w in q_weights.items():
        scale, zp = scales[name]
        shape = [q_w.size(0)] + [1] * (q_w.dim() - 1)
        s = scale.view(shape)
        zp = zp.view(shape)
        # 反量化
        sd[name] = (q_w.float() - zp) * s
        
    for name, param in other.items():
        sd[name] = param

    # 建立支援 Quantized 包裝的讀取函數
    def get_sd(name):
        if name in sd:
            return sd[name]
            
        # 如果被 QuantizedConv1d / QuantizedLinear 包裝，名稱會多出 .conv. 或 .linear.
        parts = name.split('.')
        if len(parts) >= 2:
            suffix = parts[-1]  # weight 或 bias
            base = '.'.join(parts[:-1])
            alt_conv = f"{base}.conv.{suffix}"
            if alt_conv in sd:
                return sd[alt_conv]
            alt_linear = f"{base}.linear.{suffix}"
            if alt_linear in sd:
                return sd[alt_linear]
        
        raise KeyError(f"Could not find key '{name}' or its wrapped variants in checkpoint.")

    # 建立 model.json 的 manifest 結構
    m = {
        'format': 'silentspeech_mamba_v1_qat_int8',
        'd_model': D_MODEL,
        'num_layers': NUM_LAYERS,
        'd_state': D_STATE,
        'd_conv': D_CONV,
        'd_inner': D_INNER,
        'dt_rank': DT_RANK,
        'num_outs': NUM_OUTS,
        'num_aux_outs': NUM_AUX,
        'tensors': {}
    }

    # 輸出張量輔助函式
    def export_quantized(name, tensor):
        # 進行量化
        q_tensor, scale, zp = quantize_tensor(tensor, bits)
        
        # 寫入 int8 權重檔
        fn_weight = name.replace('.', '_') + '.bin'
        fp_weight = os.path.join(OUTPUT_DIR, fn_weight)
        q_arr = q_tensor.numpy()
        q_arr.tofile(fp_weight)
        
        # 寫入 scale 檔
        fn_scale = name.replace('.', '_') + '_scale.bin'
        fp_scale = os.path.join(OUTPUT_DIR, fn_scale)
        s_arr = scale.float().numpy()
        s_arr.tofile(fp_scale)
        
        # 寫入 zero_point 檔
        fn_zp = name.replace('.', '_') + '_zp.bin'
        fp_zp = os.path.join(OUTPUT_DIR, fn_zp)
        zp_arr = zp.float().numpy()
        zp_arr.tofile(fp_zp)
        
        # 在 manifest 紀錄
        m['tensors'][name] = {
            'file': fn_weight,
            'shape': list(tensor.shape),
            'numel': int(tensor.numel()),
            'dtype': 'int8',
            'scale_file': fn_scale,
            'zp_file': fn_zp
        }
        print(f'  [INT8]  {name:50s} {str(list(tensor.shape)):20s}')

    def export_fp32(name, tensor):
        fn = name.replace('.', '_') + '.bin'
        fp = os.path.join(OUTPUT_DIR, fn)
        a = tensor.detach().cpu().contiguous().float().numpy()
        a.tofile(fp)
        
        m['tensors'][name] = {
            'file': fn,
            'shape': list(tensor.shape),
            'numel': int(tensor.numel()),
            'dtype': 'float32'
        }
        print(f'  [FP32]  {name:50s} {str(list(tensor.shape)):20s}')

    print('=== ResBlocks (BN fused & INT8 quantized) ===')
    for i in range(3):
        p = f'conv_blocks.{i*2}'
        for tag, cv, bn in [('conv1', 'conv1', 'bn1'), ('conv2', 'conv2', 'bn2')]:
            # 融合 BN
            fw, fb = fuse_bn(
                get_sd(f'{p}.{cv}.weight'), get_sd(f'{p}.{cv}.bias'),
                get_sd(f'{p}.{bn}.weight'), get_sd(f'{p}.{bn}.bias'),
                get_sd(f'{p}.{bn}.running_mean'), get_sd(f'{p}.{bn}.running_var')
            )
            # 導出
            export_quantized(f'res{i}.{tag}.weight', fw)
            export_fp32(f'res{i}.{tag}.bias', fb)
            
        if f'{p}.residual_path.weight' in sd or f'{p}.residual_path.conv.weight' in sd:
            fw, fb = fuse_bn(
                get_sd(f'{p}.residual_path.weight'), get_sd(f'{p}.residual_path.bias'),
                get_sd(f'{p}.res_norm.weight'), get_sd(f'{p}.res_norm.bias'),
                get_sd(f'{p}.res_norm.running_mean'), get_sd(f'{p}.res_norm.running_var')
            )
            export_quantized(f'res{i}.skip.weight', fw)
            export_fp32(f'res{i}.skip.bias', fb)
            m['tensors'][f'res{i}.has_skip'] = True
        else:
            m['tensors'][f'res{i}.has_skip'] = False
        m['tensors'][f'res{i}.stride'] = 2

    print('=== w_raw_in ===')
    export_quantized('w_raw_in.weight', get_sd('w_raw_in.weight'))
    export_fp32('w_raw_in.bias', get_sd('w_raw_in.bias'))

    print('=== Mamba Layers ===')
    for i in range(NUM_LAYERS):
        p = f'mamba_layers.{i}'
        # norm 保持 fp32
        export_fp32(f'mamba{i}.norm.weight', get_sd(f'{p}.norm.weight'))
        export_fp32(f'mamba{i}.norm.bias', get_sd(f'{p}.norm.bias'))
        
        # Mamba 內部投影矩陣與卷積進行量化
        export_quantized(f'mamba{i}.in_proj.weight', get_sd(f'{p}.mamba.in_proj.weight'))
        export_quantized(f'mamba{i}.conv1d.weight', get_sd(f'{p}.mamba.conv1d.weight'))
        export_fp32(f'mamba{i}.conv1d.bias', get_sd(f'{p}.mamba.conv1d.bias')) # bias 保持 fp32
        export_quantized(f'mamba{i}.x_proj.weight', get_sd(f'{p}.mamba.x_proj.weight'))
        export_quantized(f'mamba{i}.dt_proj.weight', get_sd(f'{p}.mamba.dt_proj.weight'))
        export_fp32(f'mamba{i}.dt_proj.bias', get_sd(f'{p}.mamba.dt_proj.bias')) # bias 保持 fp32
        
        # A_log 與 D 保持 fp32
        export_fp32(f'mamba{i}.A_log', get_sd(f'{p}.mamba.A_log'))
        export_fp32(f'mamba{i}.D', get_sd(f'{p}.mamba.D'))
        
        export_quantized(f'mamba{i}.out_proj.weight', get_sd(f'{p}.mamba.out_proj.weight'))

    print('=== Output layers ===')
    export_fp32('final_norm.weight', get_sd('final_norm.weight'))
    export_fp32('final_norm.bias', get_sd('final_norm.bias'))
    export_quantized('w_out.weight', get_sd('w_out.weight'))
    export_fp32('w_out.bias', get_sd('w_out.bias'))
    export_quantized('w_aux.weight', get_sd('w_aux.weight'))
    export_fp32('w_aux.bias', get_sd('w_aux.bias'))

    # === Golden test with pure-Python Mamba ===
    print('\n=== Golden test (pure-Python Mamba, CPU) ===')
    torch.manual_seed(42)
    test_input = torch.randn(1, 128, 8)
    export_fp32('golden.input', test_input.squeeze(0))
    
    # 執行與導出時相同的 BN 融合前向推論以求取 golden 數值
    def resblock_fwd(x_in, idx):
        p = f'conv_blocks.{idx*2}'
        fw1, fb1 = fuse_bn(
            get_sd(f'{p}.conv1.weight'), get_sd(f'{p}.conv1.bias'),
            get_sd(f'{p}.bn1.weight'), get_sd(f'{p}.bn1.bias'),
            get_sd(f'{p}.bn1.running_mean'), get_sd(f'{p}.bn1.running_var')
        )
        fw2, fb2 = fuse_bn(
            get_sd(f'{p}.conv2.weight'), get_sd(f'{p}.conv2.bias'),
            get_sd(f'{p}.bn2.weight'), get_sd(f'{p}.bn2.bias'),
            get_sd(f'{p}.bn2.running_mean'), get_sd(f'{p}.bn2.running_var')
        )
        inp = x_in
        x = F.relu(F.conv1d(x_in, fw1, fb1, stride=2, padding=1))
        x = F.conv1d(x, fw2, fb2, stride=1, padding=1)
        if f'{p}.residual_path.weight' in sd or f'{p}.residual_path.conv.weight' in sd:
            fws, fbs = fuse_bn(
                get_sd(f'{p}.residual_path.weight'), get_sd(f'{p}.residual_path.bias'),
                get_sd(f'{p}.res_norm.weight'), get_sd(f'{p}.res_norm.bias'),
                get_sd(f'{p}.res_norm.running_mean'), get_sd(f'{p}.res_norm.running_var')
            )
            res = F.conv1d(inp, fws, fbs, stride=2)
        else:
            res = inp
        return F.relu(x + res)

    x = test_input.transpose(1, 2)  # (1, 8, 128)
    for bi in range(3):
        x = resblock_fwd(x, bi)
        export_fp32(f'golden.after_res{bi}', x.squeeze(0).transpose(0, 1))
        
    x = x.transpose(1, 2)  # (1, T, D_MODEL)
    x = x @ get_sd('w_raw_in.weight').T + get_sd('w_raw_in.bias')
    export_fp32('golden.after_w_raw_in', x.squeeze(0))
    
    for li in range(NUM_LAYERS):
        p = f'mamba_layers.{li}'
        w = {
            'norm_w': get_sd(f'{p}.norm.weight'), 'norm_b': get_sd(f'{p}.norm.bias'),
            'in_proj_w': get_sd(f'{p}.mamba.in_proj.weight'),
            'conv1d_w': get_sd(f'{p}.mamba.conv1d.weight'), 'conv1d_b': get_sd(f'{p}.mamba.conv1d.bias'),
            'x_proj_w': get_sd(f'{p}.mamba.x_proj.weight'),
            'dt_proj_w': get_sd(f'{p}.mamba.dt_proj.weight'), 'dt_proj_b': get_sd(f'{p}.mamba.dt_proj.bias'),
            'A_log': get_sd(f'{p}.mamba.A_log'), 'D_param': get_sd(f'{p}.mamba.D'),
            'out_proj_w': get_sd(f'{p}.mamba.out_proj.weight'),
        }
        x = mamba_forward_cpu(x, w)
        export_fp32(f'golden.after_mamba{li}', x.squeeze(0))
        
    x = F.layer_norm(x, [D_MODEL], get_sd('final_norm.weight'), get_sd('final_norm.bias'))
    export_fp32('golden.after_final_norm', x.squeeze(0))
    mel = x @ get_sd('w_out.weight').T + get_sd('w_out.bias')
    aux = x @ get_sd('w_aux.weight').T + get_sd('w_aux.bias')
    export_fp32('golden.mel_output', mel.squeeze(0))
    export_fp32('golden.aux_output', aux.squeeze(0))

    # Golden Streaming
    from absl import flags
    try:
        flags.FLAGS(['export_weights_qat.py'])
    except Exception:
        pass
    try:
        flags.FLAGS.model_size = D_MODEL
        flags.FLAGS.num_layers = NUM_LAYERS
        flags.FLAGS.dropout = 0.0
    except Exception:
        pass

    sys.path.insert(0, '.')
    from architecture import Model as ArchModel
    
    # 暫時用 arch_model 做前向求值
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_st = ArchModel(num_features=8, num_outs=NUM_OUTS, num_aux_outs=NUM_AUX).to(dev)
    
    # 重構一個沒有 Quantized 包裝的單純 FP32 state_dict 給 Model 載入
    clean_sd = {}
    for k in sd.keys():
        clean_k = k.replace('.conv.', '.').replace('.linear.', '.')
        clean_sd[clean_k] = sd[k]
        
    model_st.load_state_dict(clean_sd, strict=False)
    model_st.eval()
    
    if dev == 'cuda':
        with torch.no_grad():
            st_input = test_input.to(dev)
            st_state = model_st.init_streaming_state(dev)
            mel_st, _, _ = model_st.forward_streaming(st_input, st_state)
            export_fp32('golden.mel_stream_output', mel_st.squeeze(0).cpu())
    else:
        print('  ⚠️  跳過 golden streaming (mamba_ssm.step() 需要 CUDA)')

    # 儲存 manifest
    with open(os.path.join(OUTPUT_DIR, 'model.json'), 'w') as f:
        json.dump(m, f, indent=2)
        
    print(f'\nExported successfully to {OUTPUT_DIR}/')

if __name__ == '__main__':
    main()
