#pragma once
#include "tensor.h"
#include "weight_loader.h"

// ── ResBlock weights (BN already fused) ─────────────────────────
struct ResBlockWeights {
    Tensor conv1_w, conv1_b;  // (C_out, C_in, 3)
    Tensor conv2_w, conv2_b;  // (C_out, C_out, 3)
    bool has_skip;
    Tensor skip_w, skip_b;    // (C_out, C_in, 1) if has_skip
    int stride;
};

// ── Mamba layer weights ─────────────────────────────────────────
struct MambaWeights {
    Tensor norm_w, norm_b;        // (d_model)
    Tensor in_proj_w;             // (2*d_inner, d_model)
    Tensor conv1d_w, conv1d_b;    // (d_inner, 1, d_conv), (d_inner)
    Tensor x_proj_w;              // (dt_rank + 2*d_state, d_inner)
    Tensor dt_proj_w, dt_proj_b;  // (d_inner, dt_rank), (d_inner)
    Tensor A_log;                 // (d_inner, d_state)
    Tensor D;                     // (d_inner)
    Tensor out_proj_w;            // (d_model, d_inner)
    Tensor neg_exp_A;             // pre-computed -exp(A_log), (d_inner, d_state)
    int d_model, d_inner, d_state, d_conv, dt_rank;
};

// ── Streaming State ─────────────────────────────────────────────
// Maintains all stateful information across streaming chunks.
struct StreamingState {
    // CNN causal conv caches: 3 blocks x 2 convs per block
    // cache1[i]: (C_in, K-1) where K=3 -> (C_in, 2)
    // cache2[i]: (C_out, K-1) -> (C_out, 2)
    Tensor cnn_cache1[3], cnn_cache2[3];
    bool cnn_init1[3], cnn_init2[3];

    // Mamba states: 4 layers
    // conv_state[i]: (d_inner, d_conv) — causal conv1d ring buffer
    // ssm_state[i]:  (d_inner, d_state) — selective scan hidden state
    Tensor mamba_conv_state[4];
    Tensor mamba_ssm_state[4];
    int conv_pos[4];              // ring buffer write position for mamba conv_state
};

// ── Full EMG Model ──────────────────────────────────────────────
class EmgModel {
public:
    void load(WeightMap& wm);

    // ── Batch inference ─────────────────────────────────────────
    // raw_emg (T, 8) -> mel (T/8, num_outs)
    Tensor forward(const Tensor& raw_emg);

    // Per-layer golden comparison
    struct LayerOutputs {
        Tensor after_res[3];
        Tensor after_w_raw_in;
        Tensor after_mamba[4];
        Tensor after_final_norm;
        Tensor mel_output;
        Tensor aux_output;
    };
    LayerOutputs forward_debug(const Tensor& raw_emg);

    // ── Streaming inference ─────────────────────────────────────
    // Initialize all streaming state (zero buffers)
    StreamingState init_streaming_state();

    // Process one chunk: raw_emg (chunk_len, 8) -> mel (chunk_frames, num_outs)
    // State is updated in-place across calls.
    Tensor forward_streaming(const Tensor& raw_emg, StreamingState& state);

    // Accessors
    int get_num_outs() const { return num_outs; }
    int get_d_model() const { return d_model; }

private:
    int d_model, num_layers, d_state, d_conv, d_inner, dt_rank;
    int num_outs, num_aux_outs;

    // Pre-allocated workspace for streaming Mamba step (avoids per-timestep malloc)
    struct MambaStepBuffers {
        std::vector<float> x_norm;   // (d_model)
        std::vector<float> xz;      // (2 * d_inner)
        std::vector<float> x_conv;  // (d_inner)
        std::vector<float> x_dbc;   // (dt_rank + 2 * d_state)
        std::vector<float> delta;   // (d_inner)
        std::vector<float> y;       // (d_inner)
        std::vector<float> out;     // (d_model)
    };
    MambaStepBuffers step_bufs_;

    ResBlockWeights res_blocks[3];
    Tensor w_raw_in_w, w_raw_in_b;
    MambaWeights mamba_layers[4];
    Tensor final_norm_w, final_norm_b;
    Tensor w_out_w, w_out_b;
    Tensor w_aux_w, w_aux_b;

    // ── Batch forward helpers ───────────────────────────────────
    Tensor resblock_forward(const Tensor& x, int idx);
    Tensor mamba_forward(const Tensor& x, int layer_idx);

    // ── Streaming forward helpers ───────────────────────────────
    Tensor resblock_forward_causal(const Tensor& x, int idx,
                                    Tensor& cache1, Tensor& cache2,
                                    bool& init1, bool& init2);
    Tensor mamba_forward_streaming(const Tensor& x, int layer_idx,
                                    Tensor& conv_state, Tensor& ssm_state,
                                    int& conv_pos);
};
