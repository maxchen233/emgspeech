#include "emg_model.h"
#include "ops.h"
#include <cmath>
#include <cstring>
#include <iostream>

// =====================================================================
// Weight loading
// =====================================================================

void EmgModel::load(WeightMap& wm) {
    d_model   = wm.get_int("d_model");
    num_layers = wm.get_int("num_layers");
    d_state   = wm.get_int("d_state");
    d_conv    = wm.get_int("d_conv");
    d_inner   = wm.get_int("d_inner");
    dt_rank   = wm.get_int("dt_rank");
    num_outs  = wm.get_int("num_outs");
    num_aux_outs = wm.get_int("num_aux_outs");

    std::cerr << "Model: d_model=" << d_model << " layers=" << num_layers
              << " d_state=" << d_state << " d_conv=" << d_conv
              << " d_inner=" << d_inner << " dt_rank=" << dt_rank << std::endl;

    // ResBlocks
    for (int i = 0; i < 3; i++) {
        std::string p = "res" + std::to_string(i);
        res_blocks[i].conv1_w = wm.get(p + ".conv1.weight");
        res_blocks[i].conv1_b = wm.get(p + ".conv1.bias");
        res_blocks[i].conv2_w = wm.get(p + ".conv2.weight");
        res_blocks[i].conv2_b = wm.get(p + ".conv2.bias");
        res_blocks[i].has_skip = wm.get_bool(p + ".has_skip");
        if (res_blocks[i].has_skip) {
            res_blocks[i].skip_w = wm.get(p + ".skip.weight");
            res_blocks[i].skip_b = wm.get(p + ".skip.bias");
        }
        res_blocks[i].stride = 2;
    }

    w_raw_in_w = wm.get("w_raw_in.weight");
    w_raw_in_b = wm.get("w_raw_in.bias");

    // Mamba layers
    for (int i = 0; i < num_layers; i++) {
        std::string p = "mamba" + std::to_string(i);
        auto& ml = mamba_layers[i];
        ml.norm_w     = wm.get(p + ".norm.weight");
        ml.norm_b     = wm.get(p + ".norm.bias");
        ml.in_proj_w  = wm.get(p + ".in_proj.weight");
        ml.conv1d_w   = wm.get(p + ".conv1d.weight");
        ml.conv1d_b   = wm.get(p + ".conv1d.bias");
        ml.x_proj_w   = wm.get(p + ".x_proj.weight");
        ml.dt_proj_w  = wm.get(p + ".dt_proj.weight");
        ml.dt_proj_b  = wm.get(p + ".dt_proj.bias");
        ml.A_log      = wm.get(p + ".A_log");
        ml.D          = wm.get(p + ".D");
        ml.out_proj_w = wm.get(p + ".out_proj.weight");
        // Pre-compute A = -exp(A_log) to avoid repeated exp() in forward
        ml.neg_exp_A = Tensor::zeros({d_inner, d_state});
        for (int d = 0; d < d_inner; d++)
            for (int s = 0; s < d_state; s++)
                ml.neg_exp_A(d, s) = -std::exp(ml.A_log(d, s));
        ml.d_model = d_model;
        ml.d_inner = d_inner;
        ml.d_state = d_state;
        ml.d_conv  = d_conv;
        ml.dt_rank = dt_rank;
    }

    final_norm_w = wm.get("final_norm.weight");
    final_norm_b = wm.get("final_norm.bias");
    w_out_w = wm.get("w_out.weight");
    w_out_b = wm.get("w_out.bias");
    w_aux_w = wm.get("w_aux.weight");
    w_aux_b = wm.get("w_aux.bias");

    std::cerr << "Model loaded successfully." << std::endl;
}

// =====================================================================
// Batch forward (unchanged from original)
// =====================================================================

Tensor EmgModel::resblock_forward(const Tensor& x_in, int idx) {
    auto& rb = res_blocks[idx];
    Tensor x = ops::conv1d(x_in, rb.conv1_w, rb.conv1_b, rb.stride, 1);
    ops::relu_(x);
    x = ops::conv1d(x, rb.conv2_w, rb.conv2_b, 1, 1);
    Tensor res;
    if (rb.has_skip) {
        res = ops::conv1d(x_in, rb.skip_w, rb.skip_b, rb.stride, 0);
    } else {
        res = x_in.clone();
    }
    x = ops::add(x, res);
    ops::relu_(x);
    return x;
}

Tensor EmgModel::mamba_forward(const Tensor& x_in, int layer_idx) {
    auto& ml = mamba_layers[layer_idx];
    const int T = x_in.size(0);
    const int D = ml.d_model;
    const int DI = ml.d_inner;
    const int DS = ml.d_state;
    const int DR = ml.dt_rank;

    // LayerNorm + in_proj (these use BLAS for batch, keep as-is)
    Tensor x = ops::layer_norm(x_in, ml.norm_w, ml.norm_b);
    Tensor xz = ops::linear(x, ml.in_proj_w, Tensor::zeros({2 * DI}));
    const float* xz_ptr = xz.ptr();

    // Transpose x_part directly from xz to conv layout (DI, T)
    // Eliminates: x_part tensor, separate transpose step
    Tensor x_conv_in = Tensor::zeros({DI, T});
    float* xci_ptr = x_conv_in.ptr();
    for (int t = 0; t < T; t++)
        for (int d = 0; d < DI; d++)
            xci_ptr[d * T + t] = xz_ptr[t * (2 * DI) + d];

    // Depthwise conv1d
    Tensor x_conv = ops::conv1d(x_conv_in, ml.conv1d_w, ml.conv1d_b,
                                1, ml.d_conv - 1, DI);

    // Fused transpose (DI, L) -> (T, DI) + SiLU
    // Eliminates: x_conv_trimmed tensor, separate silu_() call, x_silu transpose
    const int conv_L = x_conv.size(1);
    const float* xc_ptr = x_conv.ptr();
    Tensor x_silu = Tensor::zeros({T, DI});
    float* xs_ptr = x_silu.ptr();
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < DI; d++) {
            float v = xc_ptr[d * conv_L + t];
            xs_ptr[t * DI + d] = v / (1.0f + std::exp(-v));  // inline SiLU
        }
    }

    // x_proj: (T, DI) -> (T, DR + 2*DS)
    Tensor x_dbc = ops::linear(x_silu, ml.x_proj_w, Tensor::zeros({DR + 2 * DS}));
    const float* x_dbc_ptr = x_dbc.ptr();
    const int dbc_stride = DR + 2 * DS;

    // dt_proj: extract delta_input (T, DR) from x_dbc, project to (T, DI)
    Tensor delta_input = Tensor::zeros({T, DR});
    float* di_ptr = delta_input.ptr();
    for (int t = 0; t < T; t++)
        std::memcpy(di_ptr + t * DR, x_dbc_ptr + t * dbc_stride, DR * sizeof(float));

    Tensor delta = ops::linear(delta_input, ml.dt_proj_w, ml.dt_proj_b);
    // Inline softplus
    float* delta_ptr = delta.ptr();
    for (int i = 0; i < T * DI; i++)
        delta_ptr[i] = std::log(1.0f + std::exp(delta_ptr[i]));

    // Selective scan with pre-computed A, fused gate
    // Eliminates: A tensor recomputation, B_mat/C_mat tensors, separate silu_(z) + mul
    const float* A_ptr = ml.neg_exp_A.ptr();
    const float* D_ptr = ml.D.ptr();
    Tensor h = Tensor::zeros({DI, DS});
    float* h_ptr = h.ptr();
    Tensor y = Tensor::zeros({T, DI});
    float* y_ptr = y.ptr();

    for (int t = 0; t < T; t++) {
        const float* dt_row = delta_ptr + t * DI;
        const float* x_row = xs_ptr + t * DI;
        const float* B_row = x_dbc_ptr + t * dbc_stride + DR;
        const float* C_row = x_dbc_ptr + t * dbc_stride + DR + DS;
        float* y_row = y_ptr + t * DI;

        for (int d = 0; d < DI; d++) {
            float dt = dt_row[d];
            float x_t = x_row[d];
            float y_val = 0;
            float* h_d = h_ptr + d * DS;
            const float* A_d = A_ptr + d * DS;

            for (int s = 0; s < DS; s++) {
                float dA = std::exp(dt * A_d[s]);
                float dB = dt * B_row[s];
                h_d[s] = h_d[s] * dA + dB * x_t;
                y_val += h_d[s] * C_row[s];
            }
            y_val += D_ptr[d] * x_t;

            // Inline gate: y * silu(z)  — z read directly from xz, no separate tensor
            float z_val = xz_ptr[t * (2 * DI) + DI + d];
            float silu_z = z_val / (1.0f + std::exp(-z_val));
            y_row[d] = y_val * silu_z;
        }
    }

    // out_proj + inline residual addition (avoids clone + ops::add allocation)
    Tensor out = ops::linear(y, ml.out_proj_w, Tensor::zeros({D}));
    float* out_ptr = out.ptr();
    const float* res_ptr = x_in.ptr();
    for (int i = 0; i < T * D; i++)
        out_ptr[i] += res_ptr[i];

    return out;
}

Tensor EmgModel::forward(const Tensor& raw_emg) {
    auto outputs = forward_debug(raw_emg);
    return outputs.mel_output;
}

EmgModel::LayerOutputs EmgModel::forward_debug(const Tensor& raw_emg) {
    LayerOutputs out;
    int T = raw_emg.size(0);
    int C = raw_emg.size(1);

    Tensor x = Tensor::zeros({C, T});
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            x(c, t) = raw_emg(t, c);

    for (int i = 0; i < 3; i++) {
        x = resblock_forward(x, i);
        int C2 = x.size(0), T2 = x.size(1);
        out.after_res[i] = Tensor::zeros({T2, C2});
        for (int t = 0; t < T2; t++)
            for (int c = 0; c < C2; c++)
                out.after_res[i](t, c) = x(c, t);
    }

    int T2 = x.size(1);
    Tensor x_tc = Tensor::zeros({T2, d_model});
    for (int t = 0; t < T2; t++)
        for (int c = 0; c < d_model; c++)
            x_tc(t, c) = x(c, t);

    x_tc = ops::linear(x_tc, w_raw_in_w, w_raw_in_b);
    out.after_w_raw_in = x_tc.clone();

    for (int i = 0; i < num_layers; i++) {
        x_tc = mamba_forward(x_tc, i);
        out.after_mamba[i] = x_tc.clone();
    }

    x_tc = ops::layer_norm(x_tc, final_norm_w, final_norm_b);
    out.after_final_norm = x_tc.clone();

    out.mel_output = ops::linear(x_tc, w_out_w, w_out_b);
    out.aux_output = ops::linear(x_tc, w_aux_w, w_aux_b);

    return out;
}

// =====================================================================
// Streaming forward
// =====================================================================

StreamingState EmgModel::init_streaming_state() {
    StreamingState state;

    for (int i = 0; i < 3; i++) {
        state.cnn_init1[i] = false;
        state.cnn_init2[i] = false;
    }

    for (int i = 0; i < num_layers; i++) {
        state.mamba_conv_state[i] = Tensor::zeros({d_inner, d_conv});
        state.mamba_ssm_state[i]  = Tensor::zeros({d_inner, d_state});
        state.conv_pos[i] = d_conv - 1;  // ring buffer: first (pos+1)%DC = 0
    }

    return state;
}

Tensor EmgModel::resblock_forward_causal(const Tensor& x_in, int idx,
                                          Tensor& cache1, Tensor& cache2,
                                          bool& init1, bool& init2) {
    // Causal ResBlock forward — matches Python ResBlock.forward_causal()
    //
    // x_in: (C_in, T)
    // conv1 (causal, stride) + relu
    // conv2 (causal, stride=1)
    // residual (1x1 conv stride if has_skip)
    // output = relu(conv2_out + residual)

    auto& rb = res_blocks[idx];

    // conv1 (causal, stride=rb.stride)
    Tensor x = ops::conv1d_causal(x_in, rb.conv1_w, rb.conv1_b,
                                   rb.stride, cache1, init1);
    ops::relu_(x);

    // conv2 (causal, stride=1)
    x = ops::conv1d_causal(x, rb.conv2_w, rb.conv2_b,
                            1, cache2, init2);

    // residual
    Tensor res;
    if (rb.has_skip) {
        // 1x1 conv with stride — no padding, no cache needed
        res = ops::conv1d(x_in, rb.skip_w, rb.skip_b, rb.stride, 0);
    } else {
        res = x_in.clone();
    }

    x = ops::add(x, res);
    ops::relu_(x);
    return x;
}

Tensor EmgModel::mamba_forward_streaming(const Tensor& x_in, int layer_idx,
                                          Tensor& conv_state, Tensor& ssm_state,
                                          int& conv_pos_ref) {
    // ── Fused Streaming Mamba Step ──────────────────────────────────
    //
    // The entire per-timestep pipeline is fused into a single loop body
    // with zero intermediate Tensor allocations. All workspace is pre-allocated
    // in step_bufs_ and reused across timesteps.
    //
    // Pipeline per timestep:
    //   LayerNorm → in_proj → ring buffer conv → depthwise conv + SiLU
    //   → x_proj → dt_proj + softplus → selective scan → gate → out_proj
    //
    // x_in: (T, D), conv_state: (d_inner, d_conv), ssm_state: (d_inner, d_state)

    auto& ml = mamba_layers[layer_idx];
    const int T = x_in.size(0);
    const int D = ml.d_model;
    const int DI = ml.d_inner;
    const int DS = ml.d_state;
    const int DR = ml.dt_rank;
    const int DC = ml.d_conv;

    // Pre-allocate workspace buffers (resize is no-op after first call)
    auto& bufs = step_bufs_;
    bufs.x_norm.resize(D);
    bufs.xz.resize(2 * DI);
    bufs.x_conv.resize(DI);
    bufs.x_dbc.resize(DR + 2 * DS);
    bufs.delta.resize(DI);
    bufs.y.resize(DI);
    bufs.out.resize(D);

    // Cache raw pointers to weights (avoid repeated Tensor::ptr() calls)
    const float* neg_A       = ml.neg_exp_A.ptr();
    float*       conv_ptr    = conv_state.ptr();
    float*       ssm_ptr     = ssm_state.ptr();
    const float* norm_w_ptr  = ml.norm_w.ptr();
    const float* norm_b_ptr  = ml.norm_b.ptr();
    const float* in_proj_ptr = ml.in_proj_w.ptr();
    const float* conv1d_w_ptr= ml.conv1d_w.ptr();
    const float* conv1d_b_ptr= ml.conv1d_b.ptr();
    const float* x_proj_ptr  = ml.x_proj_w.ptr();
    const float* dt_proj_ptr = ml.dt_proj_w.ptr();
    const float* dt_bias_ptr = ml.dt_proj_b.ptr();
    const float* D_ptr       = ml.D.ptr();
    const float* out_proj_ptr= ml.out_proj_w.ptr();

    Tensor output = Tensor::zeros({T, D});
    float* output_ptr = output.ptr();
    const float* x_in_ptr = x_in.ptr();

    const int dbc_dim = DR + 2 * DS;

    for (int t = 0; t < T; t++) {
        const float* x_t_in = x_in_ptr + t * D;
        float* x_norm  = bufs.x_norm.data();
        float* xz      = bufs.xz.data();
        float* x_conv  = bufs.x_conv.data();
        float* x_dbc   = bufs.x_dbc.data();
        float* delta   = bufs.delta.data();
        float* y       = bufs.y.data();
        float* out     = bufs.out.data();

        // ── 1. Inline LayerNorm ──────────────────────────────────
        {
            float mean = 0.0f;
            for (int d = 0; d < D; d++) mean += x_t_in[d];
            mean /= D;
            float var = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = x_t_in[d] - mean;
                var += diff * diff;
            }
            float inv_std = 1.0f / std::sqrt(var / D + 1e-5f);
            for (int d = 0; d < D; d++) {
                x_norm[d] = (x_t_in[d] - mean) * inv_std * norm_w_ptr[d] + norm_b_ptr[d];
            }
        }

        // ── 2. Inline in_proj: (D) → (2*DI), no bias ────────────
        ops::matvec_small(in_proj_ptr, 2 * DI, D, x_norm, nullptr, xz);

        // x_part = xz[:DI], z = xz[DI:] — zero-copy via pointer offset
        float* x_part = xz;
        float* z_part = xz + DI;

        // ── 3. Ring buffer conv_state update ─────────────────────
        // O(DI) write instead of O(DI × DC) shift
        conv_pos_ref = (conv_pos_ref + 1) % DC;
        for (int d = 0; d < DI; d++) {
            conv_ptr[d * DC + conv_pos_ref] = x_part[d];
        }

        // ── 4. Depthwise conv + inline SiLU ─────────────────────
        for (int d = 0; d < DI; d++) {
            float sum = conv1d_b_ptr[d];
            const float* w_row = conv1d_w_ptr + d * DC;
            const float* cs_row = conv_ptr + d * DC;
            for (int k = 0; k < DC; k++) {
                sum += cs_row[(conv_pos_ref + 1 + k) % DC] * w_row[k];
            }
            x_conv[d] = sum / (1.0f + std::exp(-sum));  // fused SiLU
        }

        // ── 5. Inline x_proj: (DI) → (DR + 2*DS), no bias ──────
        ops::matvec_small(x_proj_ptr, dbc_dim, DI, x_conv, nullptr, x_dbc);

        // ── 6. Inline dt_proj + softplus ─────────────────────────
        // dt_proj: (DR) → (DI), reads only first DR elements of x_dbc
        ops::matvec_small(dt_proj_ptr, DI, DR, x_dbc, dt_bias_ptr, delta);
        for (int d = 0; d < DI; d++) {
            delta[d] = std::log(1.0f + std::exp(delta[d]));
        }

        // ── 7. Selective scan + D skip + gate ────────────────────
        const float* B_ptr = x_dbc + DR;
        const float* C_ptr = x_dbc + DR + DS;
        for (int d = 0; d < DI; d++) {
            float dt = delta[d];
            float xc = x_conv[d];
            float y_val = 0.0f;
            float* h_row = ssm_ptr + d * DS;
            const float* A_row = neg_A + d * DS;

            for (int s = 0; s < DS; s++) {
                float dA = std::exp(dt * A_row[s]);
                float dB = dt * B_ptr[s];
                h_row[s] = h_row[s] * dA + dB * xc;
                y_val += h_row[s] * C_ptr[s];
            }
            y_val += D_ptr[d] * xc;

            // Inline gate: y * silu(z) — z read from xz buffer, no copy
            float z_val = z_part[d];
            y[d] = y_val * (z_val / (1.0f + std::exp(-z_val)));
        }

        // ── 8. Inline out_proj + residual ────────────────────────
        ops::matvec_small(out_proj_ptr, D, DI, y, nullptr, out);
        float* out_row = output_ptr + t * D;
        for (int d = 0; d < D; d++) {
            out_row[d] = out[d] + x_t_in[d];  // residual added inline
        }
    }

    return output;
}

Tensor EmgModel::forward_streaming(const Tensor& raw_emg, StreamingState& state) {
    // Streaming forward: process one EMG chunk with maintained state.
    //
    // raw_emg: (chunk_len, 8)
    // state: updated in-place
    // output: (chunk_frames, num_outs)

    int T = raw_emg.size(0);
    int C = raw_emg.size(1);

    // Transpose (T, 8) -> (8, T) for conv
    Tensor x = Tensor::zeros({C, T});
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            x(c, t) = raw_emg(t, c);

    // Causal CNN frontend (3 ResBlocks, stride=2 each -> 8x downsampling)
    for (int i = 0; i < 3; i++) {
        x = resblock_forward_causal(x, i,
                                     state.cnn_cache1[i], state.cnn_cache2[i],
                                     state.cnn_init1[i], state.cnn_init2[i]);
    }

    // Transpose (C_out, T') -> (T', C_out)
    int T2 = x.size(1);
    int C2 = x.size(0);
    Tensor x_tc = Tensor::zeros({T2, C2});
    for (int t = 0; t < T2; t++)
        for (int c = 0; c < C2; c++)
            x_tc(t, c) = x(c, t);

    // w_raw_in
    x_tc = ops::linear(x_tc, w_raw_in_w, w_raw_in_b);

    // Mamba layers (fused step-by-step processing)
    for (int i = 0; i < num_layers; i++) {
        x_tc = mamba_forward_streaming(x_tc, i,
                                        state.mamba_conv_state[i],
                                        state.mamba_ssm_state[i],
                                        state.conv_pos[i]);
    }

    // Final norm + output head
    x_tc = ops::layer_norm(x_tc, final_norm_w, final_norm_b);
    Tensor mel = ops::linear(x_tc, w_out_w, w_out_b);

    return mel;
}
