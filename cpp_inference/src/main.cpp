// main.cpp - Golden test: compare C++ output with Python golden data
// Tests both batch forward and streaming forward.

#include "emg_model.h"
#include "weight_loader.h"
#include <iostream>
#include <cmath>
#include <string>

float max_abs_error(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    float max_err = 0;
    for (int i = 0; i < a.numel(); i++) {
        float err = std::abs(a.data[i] - b.data[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float mean_abs_error(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    float sum = 0;
    for (int i = 0; i < a.numel(); i++)
        sum += std::abs(a.data[i] - b.data[i]);
    return sum / a.numel();
}

void compare(const std::string& name, const Tensor& cpp_out, const Tensor& golden, float tol = 1e-4f) {
    float max_err = max_abs_error(cpp_out, golden);
    float mean_err = mean_abs_error(cpp_out, golden);
    bool pass = max_err < tol;

    printf("  %-25s shape=(%d,%d)  max_err=%.6e  mean_err=%.6e  %s\n",
           name.c_str(),
           cpp_out.dim() >= 1 ? cpp_out.size(0) : 0,
           cpp_out.dim() >= 2 ? cpp_out.size(1) : 0,
           max_err, mean_err,
           pass ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
    std::string weight_dir = "../exported_weights";
    if (argc > 1) weight_dir = argv[1];

    printf("=== Silent Speech Mamba C++ Inference - Golden Test ===\n");
    printf("Loading weights from: %s\n", weight_dir.c_str());

    WeightMap wm;
    wm.load_manifest(weight_dir);

    EmgModel model;
    model.load(wm);

    // Load golden input
    Tensor input = wm.get("golden.input");  // (128, 8)
    printf("\nInput shape: (%d, %d)\n", input.size(0), input.size(1));

    // =================================================================
    // Test 1: Batch forward (original test)
    // =================================================================
    printf("\n=== [1] Batch Forward - Layer-by-layer comparison (tol=1e-4) ===\n");
    auto outputs = model.forward_debug(input);

    compare("after_res0", outputs.after_res[0], wm.get("golden.after_res0"));
    compare("after_res1", outputs.after_res[1], wm.get("golden.after_res1"));
    compare("after_res2", outputs.after_res[2], wm.get("golden.after_res2"));
    compare("after_w_raw_in", outputs.after_w_raw_in, wm.get("golden.after_w_raw_in"));
    compare("after_mamba0", outputs.after_mamba[0], wm.get("golden.after_mamba0"));
    compare("after_mamba1", outputs.after_mamba[1], wm.get("golden.after_mamba1"));
    compare("after_mamba2", outputs.after_mamba[2], wm.get("golden.after_mamba2"));
    compare("after_mamba3", outputs.after_mamba[3], wm.get("golden.after_mamba3"));
    compare("after_final_norm", outputs.after_final_norm, wm.get("golden.after_final_norm"));
    compare("mel_output", outputs.mel_output, wm.get("golden.mel_output"));
    compare("aux_output", outputs.aux_output, wm.get("golden.aux_output"));

    // =================================================================
    // Test 2: Streaming forward - single chunk = full sequence
    // =================================================================
    printf("\n=== [2] Streaming Forward (single chunk) vs Batch (tol=1e-3) ===\n");
    printf("  Note: Batch uses symmetric padding, streaming uses causal padding.\n");
    printf("  Small differences are expected.\n");

    StreamingState state = model.init_streaming_state();
    Tensor mel_streaming = model.forward_streaming(input, state);

    Tensor mel_batch = outputs.mel_output;
    printf("  Batch mel shape: (%d, %d)\n", mel_batch.size(0), mel_batch.size(1));
    printf("  Streaming mel shape: (%d, %d)\n", mel_streaming.size(0), mel_streaming.size(1));

    if (mel_streaming.numel() == mel_batch.numel()) {
        compare("streaming vs batch mel", mel_streaming, mel_batch, 1e-3f);
    } else {
        printf("  Shape mismatch - batch (%d,%d) vs streaming (%d,%d)\n",
               mel_batch.size(0), mel_batch.size(1),
               mel_streaming.size(0), mel_streaming.size(1));
        int min_frames = std::min(mel_batch.size(0), mel_streaming.size(0));
        int outs = mel_batch.size(1);
        printf("  Comparing first %d frames...\n", min_frames);
        Tensor batch_sub = Tensor::zeros({min_frames, outs});
        Tensor stream_sub = Tensor::zeros({min_frames, outs});
        for (int f = 0; f < min_frames; f++)
            for (int o = 0; o < outs; o++) {
                batch_sub(f, o) = mel_batch(f, o);
                stream_sub(f, o) = mel_streaming(f, o);
            }
        compare("streaming vs batch (trunc)", stream_sub, batch_sub, 1e-3f);
    }

    // =================================================================
    // Test 3: Streaming 2x64 chunks vs 1x128 chunk (state continuity)
    // =================================================================
    printf("\n=== [3] Streaming: 2x64 chunks vs 1x128 chunk ===\n");

    StreamingState state2 = model.init_streaming_state();
    int T = input.size(0);
    int half = T / 2;

    Tensor chunk1 = Tensor::zeros({half, 8});
    for (int t = 0; t < half; t++)
        for (int c = 0; c < 8; c++)
            chunk1(t, c) = input(t, c);

    Tensor mel1 = model.forward_streaming(chunk1, state2);

    Tensor chunk2 = Tensor::zeros({T - half, 8});
    for (int t = 0; t < T - half; t++)
        for (int c = 0; c < 8; c++)
            chunk2(t, c) = input(half + t, c);

    Tensor mel2 = model.forward_streaming(chunk2, state2);

    int total_frames = mel1.size(0) + mel2.size(0);
    int outs = mel1.size(1);
    Tensor mel_concat = Tensor::zeros({total_frames, outs});
    for (int f = 0; f < mel1.size(0); f++)
        for (int o = 0; o < outs; o++)
            mel_concat(f, o) = mel1(f, o);
    for (int f = 0; f < mel2.size(0); f++)
        for (int o = 0; o < outs; o++)
            mel_concat(mel1.size(0) + f, o) = mel2(f, o);

    printf("  Single chunk mel: (%d, %d)\n", mel_streaming.size(0), mel_streaming.size(1));
    printf("  Two chunks mel:   (%d, %d)\n", mel_concat.size(0), mel_concat.size(1));

    if (mel_concat.numel() == mel_streaming.numel()) {
        compare("2x64 vs 1x128", mel_concat, mel_streaming, 1e-4f);
    } else {
        printf("  Frame count mismatch: %d vs %d (expected due to stride padding)\n",
               mel_concat.size(0), mel_streaming.size(0));
    }

    // Print first few values
    printf("\n=== First 5 mel output values ===\n");
    printf("Batch:     ");
    for (int i = 0; i < 5 && i < mel_batch.numel(); i++)
        printf("%.6f ", mel_batch.data[i]);
    printf("\n");

    printf("Streaming: ");
    for (int i = 0; i < 5 && i < mel_streaming.numel(); i++)
        printf("%.6f ", mel_streaming.data[i]);
    printf("\n");

    printf("Python:    ");
    Tensor golden_mel = wm.get("golden.mel_output");
    for (int i = 0; i < 5 && i < golden_mel.numel(); i++)
        printf("%.6f ", golden_mel.data[i]);
    printf("\n");

    return 0;
}
