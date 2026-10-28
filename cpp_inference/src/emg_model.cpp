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

    // ResBlocks — supports structurally-pruned internal widths.
    // Example:
    //   block0: 8   -> 136 -> 512
    //   block1: 512 -> 320 -> 512
    //   block2: 512 -> 328 -> 512
    for (int i = 0; i < 3; i++) {
        std::string p = "res" + std::to_string(i);
        auto& rb = res_blocks[i];

        rb.conv1_w = wm.get(p + ".conv1.weight");
        rb.conv1_b = wm.get(p + ".conv1.bias");
        rb.conv2_w = wm.get(p + ".conv2.weight");
        rb.conv2_b = wm.get(p + ".conv2.bias");
        rb.has_skip = wm.get_bool(p + ".has_skip");

        if (rb.conv1_w.dim() != 3 || rb.conv2_w.dim() != 3) {
            std::cerr << "ERROR: " << p
                      << " conv weights must be 3-D Conv1d tensors." << std::endl;
            std::exit(1);
        }

        rb.c_in  = rb.conv1_w.size(1);
        rb.c_mid = rb.conv1_w.size(0);
        rb.c_out = rb.conv2_w.size(0);

        // Critical pruning compatibility check:
        // conv2 input must equal pruned conv1 output.
        if (rb.conv2_w.size(1) != rb.c_mid) {
            std::cerr << "ERROR: " << p
                      << " shape mismatch: conv1 out=" << rb.c_mid
                      << " but conv2 in=" << rb.conv2_w.size(1)
                      << std::endl;
            std::exit(1);
        }

        if (rb.c_out != d_model) {
            std::cerr << "ERROR: " << p
                      << " output channels=" << rb.c_out
                      << " but d_model=" << d_model
                      << ". CNN->Mamba interface must remain d_model."
                      << std::endl;
            std::exit(1);
        }

        int expected_in = (i == 0) ? 8 : d_model;
        if (rb.c_in != expected_in) {
            std::cerr << "ERROR: " << p
                      << " input channels=" << rb.c_in
                      << ", expected " << expected_in << std::endl;
            std::exit(1);
        }

        if (rb.conv1_b.numel() != rb.c_mid ||
            rb.conv2_b.numel() != rb.c_out) {
            std::cerr << "ERROR: " << p
                      << " fused bias shape mismatch." << std::endl;
            std::exit(1);
        }

        if (rb.has_skip) {
            rb.skip_w = wm.get(p + ".skip.weight");
            rb.skip_b = wm.get(p + ".skip.bias");

            if (rb.skip_w.dim() != 3 ||
                rb.skip_w.size(0) != rb.c_out ||
                rb.skip_w.size(1) != rb.c_in ||
                rb.skip_w.size(2) != 1) {
                std::cerr << "ERROR: " << p
                          << " skip shape mismatch." << std::endl;
                std::exit(1);
            }
        }

        rb.stride = 2;

        std::cerr << "  " << p << ": "
                  << rb.c_in << " -> "
                  << rb.c_mid << " -> "
                  << rb.c_out << std::endl;
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

        // A is constant during inference, so compute -exp(A_log) once at load
        // time instead of once per layer per streaming chunk.
        Tensor A_log = wm.get(p + ".A_log");
        ml.A = Tensor::zeros({d_inner, d_state});
        for (int d = 0; d < d_inner; ++d) {
            for (int st = 0; st < d_state; ++st) {
                ml.A(d, st) = -std::exp(A_log(d, st));
            }
        }

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
    if (num_aux_outs > 0) {
        w_aux_w = wm.get("w_aux.weight");
        w_aux_b = wm.get("w_aux.bias");
    }

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
    ops::add_(x, res);
    ops::relu_(x);
    return x;
}

Tensor EmgModel::mamba_forward(const Tensor& x_in, int layer_idx) {
    auto& ml = mamba_layers[layer_idx];
<<<<<<< Updated upstream
    const int T = x_in.size(0);
    const int D = ml.d_model;
    const int DI = ml.d_inner;
    const int DS = ml.d_state;
    const int DR = ml.dt_rank;

    // LayerNorm + in_proj (these use BLAS for batch, keep as-is)
    Tensor x = ops::layer_norm(x_in, ml.norm_w, ml.norm_b);
    Tensor xz = ops::linear(x, ml.in_proj_w, Tensor::zeros({2 * DI}));
    const float* xz_ptr = xz.ptr();
=======
    int T = x_in.size(0);
    int DI = ml.d_inner;
    int DS = ml.d_state;
    int DR = ml.dt_rank;

    Tensor x = ops::layer_norm(x_in, ml.norm_w, ml.norm_b);
    Tensor xz = ops::linear_no_bias(x, ml.in_proj_w);
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
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

=======
    for (int t = 0; t < T; t++)
        for (int d = 0; d < DI; d++)
            x_silu(t, d) = x_conv_trimmed(d, t);

    Tensor x_dbc = ops::linear_no_bias(x_silu, ml.x_proj_w);

    Tensor delta = ops::linear_prefix(x_dbc, DR, ml.dt_proj_w, ml.dt_proj_b);
    ops::softplus_(delta);

    Tensor B_mat = Tensor::zeros({T, DS});
    Tensor C_mat = Tensor::zeros({T, DS});
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < DS; s++) {
            B_mat(t, s) = x_dbc(t, DR + s);
            C_mat(t, s) = x_dbc(t, DR + DS + s);
        }
    }

    Tensor h = Tensor::zeros({DI, DS});
    Tensor y = Tensor::zeros({T, DI});

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < DI; d++) {
            float dt = delta(t, d);
            float x_t = x_silu(t, d);
            float y_val = 0;

            for (int s = 0; s < DS; s++) {
                float dA = std::exp(dt * ml.A(d, s));
                float dB = dt * B_mat(t, s);
                h(d, s) = h(d, s) * dA + dB * x_t;
                y_val += h(d, s) * C_mat(t, s);
            }

            y_val += ml.D(d) * x_t;
            y(t, d) = y_val;
        }
    }

    ops::silu_(z);
    ops::mul_(y, z);
    Tensor out = ops::linear_no_bias(y, ml.out_proj_w);
    ops::add_(out, x_in);
>>>>>>> Stashed changes
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
    if (num_aux_outs > 0) {
        out.aux_output = ops::linear(x_tc, w_aux_w, w_aux_b);
    } else {
        out.aux_output = Tensor();
    }

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

    ops::add_(x, res);
    ops::relu_(x);
    return x;
}

Tensor EmgModel::mamba_forward_streaming(const Tensor& x_in, int layer_idx,
<<<<<<< Updated upstream
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
=======
                                          Tensor& conv_state, Tensor& ssm_state) {
    // Optimized streaming Mamba.
    //
    // The projections are independent across time, so compute them for the
    // whole chunk with SGEMM. Only the depthwise causal convolution and the
    // selective scan keep their recurrent state across timesteps.
    //
    // x_in:      (T, d_model)
    // conv_state:(d_inner, d_conv)
    // ssm_state: (d_inner, d_state)
    // output:    (T, d_model)

    auto& ml = mamba_layers[layer_idx];
    const int T  = x_in.size(0);
>>>>>>> Stashed changes
    const int DI = ml.d_inner;
    const int DS = ml.d_state;
    const int DR = ml.dt_rank;
    const int DC = ml.d_conv;
<<<<<<< Updated upstream

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
=======
    const int DBC = DR + 2 * DS;

    // 1) LayerNorm for the whole chunk.
    Tensor x_norm = ops::layer_norm(x_in, ml.norm_w, ml.norm_b);

    // 2) in_proj for all frames at once: (T,D) -> (T,2*DI).
    Tensor xz = ops::linear_no_bias(x_norm, ml.in_proj_w);

    // 3) Stateful depthwise causal convolution.
    // Process one channel at a time across the entire chunk. Channels are
    // independent, making this loop safe to parallelize with OpenMP.
    Tensor x_conv = Tensor::zeros({T, DI});

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(DI >= 256)
    #endif
    for (int d = 0; d < DI; ++d) {
        float* state_row = conv_state.ptr() + static_cast<size_t>(d) * DC;
        const float* w_row = ml.conv1d_w.ptr() + static_cast<size_t>(d) * DC;
        const float bias = ml.conv1d_b.data[d];

        for (int t = 0; t < T; ++t) {
            // Equivalent to the original shift-left + insert x_part.
            for (int k = 0; k < DC - 1; ++k) {
                state_row[k] = state_row[k + 1];
            }
            state_row[DC - 1] = xz(t, d);  // x_part = first DI columns

            float v = bias;
            for (int k = 0; k < DC; ++k) {
                v += state_row[k] * w_row[k];
>>>>>>> Stashed changes
            }
            y_val += D_ptr[d] * xc;

<<<<<<< Updated upstream
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
=======
            // SiLU in-place while the value is hot in a register.
            x_conv(t, d) = v / (1.0f + std::exp(-v));
        }
    }

    // 4) x_proj for all frames: (T,DI) -> (T,DR+2*DS).
    Tensor x_dbc = ops::linear_no_bias(x_conv, ml.x_proj_w);

    // 5) dt_proj only needs x_dbc[:, :DR]. linear_prefix uses the original
    // row stride directly, avoiding a separate delta_input allocation/copy.
    Tensor delta = ops::linear_prefix(x_dbc, DR, ml.dt_proj_w, ml.dt_proj_b);
    ops::softplus_(delta);

    // 6) Stateful selective scan. Different d_inner channels are independent,
    // so parallelize over d while each thread walks time in order.
    Tensor y = Tensor::zeros({T, DI});

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(DI >= 256)
    #endif
    for (int d = 0; d < DI; ++d) {
        float* h_row = ssm_state.ptr() + static_cast<size_t>(d) * DS;
        const float* A_row = ml.A.ptr() + static_cast<size_t>(d) * DS;
        const float D_d = ml.D.data[d];

        for (int t = 0; t < T; ++t) {
            const float dt = delta(t, d);
            const float xc = x_conv(t, d);
            const float* dbc_row = x_dbc.ptr() + static_cast<size_t>(t) * DBC;
            const float* B_row = dbc_row + DR;
            const float* C_row = B_row + DS;

            float y_val = 0.0f;
            for (int st = 0; st < DS; ++st) {
                const float dA = std::exp(dt * A_row[st]);
                h_row[st] = h_row[st] * dA + (dt * B_row[st]) * xc;
                y_val += h_row[st] * C_row[st];
            }

            y_val += D_d * xc;

            // Gate with silu(z). z is the second half of xz, so no split tensor
            // is needed.
            const float z = xz(t, DI + d);
            const float z_silu = z / (1.0f + std::exp(-z));
            y(t, d) = y_val * z_silu;
        }
    }

    // 7) out_proj for all frames, then residual add in place.
    Tensor out = ops::linear_no_bias(y, ml.out_proj_w);
    ops::add_(out, x_in);
    return out;
>>>>>>> Stashed changes
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
