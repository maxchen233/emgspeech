#include "emg_model.h"
#include "ops.h"
#include <cmath>
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
    int T = x_in.size(0);
    int D = ml.d_model;
    int DI = ml.d_inner;
    int DS = ml.d_state;
    int DR = ml.dt_rank;

    Tensor residual = x_in.clone();

    Tensor x = ops::layer_norm(x_in, ml.norm_w, ml.norm_b);
    Tensor xz = ops::linear(x, ml.in_proj_w, Tensor::zeros({2 * DI}));

    Tensor x_part = Tensor::zeros({T, DI});
    Tensor z = Tensor::zeros({T, DI});
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < DI; d++) {
            x_part(t, d) = xz(t, d);
            z(t, d) = xz(t, d + DI);
        }
    }

    Tensor x_conv_in = Tensor::zeros({DI, T});
    for (int t = 0; t < T; t++)
        for (int d = 0; d < DI; d++)
            x_conv_in(d, t) = x_part(t, d);

    Tensor x_conv = ops::conv1d(x_conv_in, ml.conv1d_w, ml.conv1d_b,
                                1, ml.d_conv - 1, DI);
    Tensor x_conv_trimmed = Tensor::zeros({DI, T});
    for (int d = 0; d < DI; d++)
        for (int t = 0; t < T; t++)
            x_conv_trimmed(d, t) = x_conv(d, t);

    ops::silu_(x_conv_trimmed);

    Tensor x_silu = Tensor::zeros({T, DI});
    for (int t = 0; t < T; t++)
        for (int d = 0; d < DI; d++)
            x_silu(t, d) = x_conv_trimmed(d, t);

    Tensor x_dbc = ops::linear(x_silu, ml.x_proj_w, Tensor::zeros({DR + 2 * DS}));

    Tensor delta_input = Tensor::zeros({T, DR});
    for (int t = 0; t < T; t++)
        for (int r = 0; r < DR; r++)
            delta_input(t, r) = x_dbc(t, r);

    Tensor delta = ops::linear(delta_input, ml.dt_proj_w, ml.dt_proj_b);
    ops::softplus_(delta);

    Tensor B_mat = Tensor::zeros({T, DS});
    Tensor C_mat = Tensor::zeros({T, DS});
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < DS; s++) {
            B_mat(t, s) = x_dbc(t, DR + s);
            C_mat(t, s) = x_dbc(t, DR + DS + s);
        }
    }

    Tensor A = Tensor::zeros({DI, DS});
    for (int d = 0; d < DI; d++)
        for (int s = 0; s < DS; s++)
            A(d, s) = -std::exp(ml.A_log(d, s));

    Tensor h = Tensor::zeros({DI, DS});
    Tensor y = Tensor::zeros({T, DI});

    for (int t = 0; t < T; t++) {
        for (int d = 0; d < DI; d++) {
            float dt = delta(t, d);
            float x_t = x_silu(t, d);
            float y_val = 0;

            for (int s = 0; s < DS; s++) {
                float dA = std::exp(dt * A(d, s));
                float dB = dt * B_mat(t, s);
                h(d, s) = h(d, s) * dA + dB * x_t;
                y_val += h(d, s) * C_mat(t, s);
            }

            y_val += ml.D(d) * x_t;
            y(t, d) = y_val;
        }
    }

    ops::silu_(z);
    y = ops::mul(y, z);
    Tensor out = ops::linear(y, ml.out_proj_w, Tensor::zeros({D}));
    return ops::add(residual, out);
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
                                          Tensor& conv_state, Tensor& ssm_state) {
    // Streaming Mamba forward — step-by-step processing.
    //
    // Matches Python: MambaBlock.forward_streaming() -> mamba.step()
    //
    // x_in: (T, D)
    // conv_state: (d_inner, d_conv) — updated in-place
    // ssm_state:  (d_inner, d_state) — updated in-place
    // output: (T, D)

    auto& ml = mamba_layers[layer_idx];
    int T = x_in.size(0);
    int D = ml.d_model;
    int DI = ml.d_inner;
    int DS = ml.d_state;
    int DR = ml.dt_rank;
    int DC = ml.d_conv;

    Tensor residual = x_in.clone();

    // Pre-compute A = -exp(A_log)
    Tensor A = Tensor::zeros({DI, DS});
    for (int d = 0; d < DI; d++)
        for (int s = 0; s < DS; s++)
            A(d, s) = -std::exp(ml.A_log(d, s));

    Tensor output = Tensor::zeros({T, D});

    // Process each timestep (mamba.step equivalent)
    for (int t = 0; t < T; t++) {

        // 1. Extract x_t from x_in
        Tensor x_t = Tensor::zeros({D});
        for (int d = 0; d < D; d++)
            x_t.data[d] = x_in(t, d);

        // 2. LayerNorm on single vector
        x_t = ops::layer_norm_1d(x_t, ml.norm_w, ml.norm_b);

        // 3. in_proj: (D) -> (2*DI)
        Tensor xz = ops::linear_1d(x_t, ml.in_proj_w, Tensor::zeros({2 * DI}));

        // Split: x_part = xz[:DI], z = xz[DI:]
        Tensor x_part = Tensor::zeros({DI});
        Tensor z = Tensor::zeros({DI});
        for (int d = 0; d < DI; d++) {
            x_part.data[d] = xz.data[d];
            z.data[d] = xz.data[d + DI];
        }

        // 4. Update conv_state: shift left, insert x_part at right
        for (int d = 0; d < DI; d++) {
            for (int k = 0; k < DC - 1; k++) {
                conv_state(d, k) = conv_state(d, k + 1);
            }
            conv_state(d, DC - 1) = x_part.data[d];
        }

        // 5. Depthwise conv1d on conv_state
        //    conv1d_w: (DI, 1, DC) — depthwise (groups=DI)
        Tensor x_conv = Tensor::zeros({DI});
        for (int d = 0; d < DI; d++) {
            float sum = ml.conv1d_b.data[d];
            for (int k = 0; k < DC; k++) {
                sum += conv_state(d, k) * ml.conv1d_w(d, 0, k);
            }
            x_conv.data[d] = sum;
        }

        // 6. SiLU
        ops::silu_(x_conv);

        // 7. x_proj: (DI) -> (DR + 2*DS)
        Tensor x_dbc = ops::linear_1d(x_conv, ml.x_proj_w,
                                       Tensor::zeros({DR + 2 * DS}));

        // 8. dt_proj on first DR elements: (DR) -> (DI)
        Tensor delta_input = Tensor::zeros({DR});
        for (int r = 0; r < DR; r++)
            delta_input.data[r] = x_dbc.data[r];

        Tensor delta = ops::linear_1d(delta_input, ml.dt_proj_w, ml.dt_proj_b);
        ops::softplus_(delta);

        // 9. Selective scan step
        Tensor y = Tensor::zeros({DI});
        for (int d = 0; d < DI; d++) {
            float dt = delta.data[d];
            float xc = x_conv.data[d];
            float y_val = 0;

            for (int s = 0; s < DS; s++) {
                float dA = std::exp(dt * A(d, s));
                float B_s = x_dbc.data[DR + s];
                float dB = dt * B_s;
                ssm_state(d, s) = ssm_state(d, s) * dA + dB * xc;

                float C_s = x_dbc.data[DR + DS + s];
                y_val += ssm_state(d, s) * C_s;
            }

            y_val += ml.D.data[d] * xc;
            y.data[d] = y_val;
        }

        // 10. Gate: y = y * silu(z)
        ops::silu_(z);
        y = ops::mul(y, z);

        // 11. out_proj: (DI) -> (D)
        Tensor out = ops::linear_1d(y, ml.out_proj_w, Tensor::zeros({D}));

        // 12. Store (residual added at end)
        for (int d = 0; d < D; d++)
            output(t, d) = out.data[d];
    }

    return ops::add(residual, output);
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

    // Mamba layers (step-by-step)
    for (int i = 0; i < num_layers; i++) {
        x_tc = mamba_forward_streaming(x_tc, i,
                                        state.mamba_conv_state[i],
                                        state.mamba_ssm_state[i]);
    }

    // Final norm + output head
    x_tc = ops::layer_norm(x_tc, final_norm_w, final_norm_b);
    Tensor mel = ops::linear(x_tc, w_out_w, w_out_b);

    return mel;
}
