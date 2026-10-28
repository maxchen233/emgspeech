#include "ops.h"
#include <cmath>
#include <cassert>
#include <cblas.h>

namespace ops {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // input: (T, D_in), weight: (D_out, D_in), bias: (D_out)
    // output: (T, D_out) = input × weight^T + bias
    int T = input.size(0);
    int D_in = input.size(1);
    int D_out = weight.size(0);
    assert(weight.size(1) == D_in);

    Tensor out = Tensor::zeros({T, D_out});

    // Pre-fill output rows with bias
    for (int t = 0; t < T; t++) {
        float* out_row = out.ptr() + t * D_out;
        for (int j = 0; j < D_out; j++) {
            out_row[j] = bias(j);
        }
    }

    // out += input * weight^T
    // input: (T, D_in) row-major  => M=T, K=D_in, lda=D_in
    // weight: (D_out, D_in) row-major, transposed => N=D_out, ldb=D_in
    // out: (T, D_out) row-major   => ldc=D_out
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, D_out, D_in,
                1.0f,                       // alpha
                input.ptr(), D_in,          // A, lda
                weight.ptr(), D_in,         // B, ldb (transposed)
                1.0f,                       // beta (accumulate onto bias)
                out.ptr(), D_out);          // C, ldc

    return out;
}

Tensor linear_1d(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // input: (D_in), weight: (D_out, D_in), bias: (D_out)
    // output: (D_out) = weight * input + bias   (matrix-vector multiply)
    int D_in = input.size(0);
    int D_out = weight.size(0);
    assert(weight.size(1) == D_in);

    Tensor out = Tensor::zeros({D_out});

    // Copy bias into output first
    for (int j = 0; j < D_out; j++) {
        out.data[j] = bias(j);
    }

    // out = 1.0 * weight * input + 1.0 * out(bias)
    // weight: (D_out, D_in) row-major => M=D_out, N=D_in, lda=D_in
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                D_out, D_in,
                1.0f,                       // alpha
                weight.ptr(), D_in,         // A, lda
                input.ptr(), 1,             // x, incx
                1.0f,                       // beta (accumulate onto bias)
                out.ptr(), 1);              // y, incy

    return out;
}

Tensor conv1d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride, int padding, int groups, int dilation) {
    // input: (C_in, L)
    // weight: (C_out, C_in/groups, K)
    // output: (C_out, L_out)
    // With dilation: L_out = (L + 2*padding - dilation*(K-1) - 1) / stride + 1
    int C_in = input.size(0);
    int L = input.size(1);
    int C_out = weight.size(0);
    int K = weight.size(2);
    int C_in_per_group = C_in / groups;

    int L_out = (L + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    Tensor out = Tensor::zeros({C_out, L_out});

    int channels_per_group_out = C_out / groups;

    // Use im2col + sgemm when groups==1, K>1, and dilation==1 (standard conv)
    // For dilated, depthwise (groups>1), or pointwise (K==1), use loop-based
    if (groups == 1 && K > 1 && dilation == 1) {
        // im2col: unfold input into (L_out, C_in * K)
        int col_rows = L_out;
        int col_cols = C_in * K;
        std::vector<float> col(col_rows * col_cols, 0.0f);

        for (int ol = 0; ol < L_out; ol++) {
            for (int ci = 0; ci < C_in; ci++) {
                for (int ki = 0; ki < K; ki++) {
                    int il = ol * stride + ki - padding;
                    if (il >= 0 && il < L) {
                        col[ol * col_cols + ci * K + ki] = input(ci, il);
                    }
                    // else: stays 0 (zero-padding)
                }
            }
        }

        // weight reshaped as (C_out, C_in*K), already contiguous in memory
        // out_T = col * weight^T  =>  (L_out, C_out)
        // Then we transpose to (C_out, L_out)
        std::vector<float> out_T(L_out * C_out, 0.0f);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    L_out, C_out, col_cols,
                    1.0f,
                    col.data(), col_cols,         // A = col, lda
                    weight.ptr(), col_cols,       // B = weight, ldb (transposed)
                    0.0f,
                    out_T.data(), C_out);         // C = out_T, ldc

        // Transpose (L_out, C_out) -> (C_out, L_out) and add bias
        for (int co = 0; co < C_out; co++) {
            float b = bias(co);
            for (int ol = 0; ol < L_out; ol++) {
                out(co, ol) = out_T[ol * C_out + co] + b;
            }
        }
    } else {
        // Loop-based implementation (handles dilation, depthwise, pointwise)
        for (int g = 0; g < groups; g++) {
            int c_out_start = g * channels_per_group_out;
            int c_in_start = g * C_in_per_group;

            for (int co = c_out_start; co < c_out_start + channels_per_group_out; co++) {
                for (int ol = 0; ol < L_out; ol++) {
                    float sum = bias(co);
                    for (int ci = 0; ci < C_in_per_group; ci++) {
                        for (int ki = 0; ki < K; ki++) {
                            int il = ol * stride + ki * dilation - padding;
                            if (il >= 0 && il < L) {
                                sum += input(c_in_start + ci, il) *
                                       weight(co, ci, ki);
                            }
                        }
                    }
                    out(co, ol) = sum;
                }
            }
        }
    }
    return out;
}

Tensor conv_transpose1d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                        int stride, int padding) {
    // input: (C_in, L)
    // weight: (C_in, C_out, K)  — note: PyTorch ConvTranspose1d weight shape
    // output: (C_out, L_out)
    // L_out = (L - 1) * stride - 2 * padding + K
    int C_in = input.size(0);
    int L = input.size(1);
    int C_out = weight.size(1);
    int K = weight.size(2);
    int L_out = (L - 1) * stride - 2 * padding + K;

    Tensor out = Tensor::zeros({C_out, L_out});

    // Add bias
    for (int co = 0; co < C_out; co++) {
        float b = bias(co);
        for (int ol = 0; ol < L_out; ol++) {
            out(co, ol) = b;
        }
    }

    // Scatter: for each input position, scatter to output positions
    for (int ci = 0; ci < C_in; ci++) {
        for (int il = 0; il < L; il++) {
            float x_val = input(ci, il);
            if (x_val == 0.0f) continue;  // skip zeros for efficiency

            for (int k = 0; k < K; k++) {
                int ol = il * stride + k - padding;
                if (ol >= 0 && ol < L_out) {
                    for (int co = 0; co < C_out; co++) {
                        out(co, ol) += x_val * weight(ci, co, k);
                    }
                }
            }
        }
    }

    return out;
}

Tensor conv1d_causal(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int stride, Tensor& cache, bool& initialized, int groups) {
    // Causal Conv1d with cache for streaming.
    //
    // input:  (C_in, T)     — new chunk
    // weight: (C_out, C_in/groups, K)
    // cache:  (C_in, K-1)   — left-context from previous chunk
    //
    // Strategy:
    //   padded = cat(cache, input) along time axis   => (C_in, K-1 + T)
    //   output = conv1d(padded, weight, bias, stride, padding=0, groups)
    //   cache  = padded[:, -(K-1):]   (update for next call)
    //
    // This produces causal output: each output sample only depends on
    // current and past input samples.

    int C_in = input.size(0);
    int T = input.size(1);
    int K = weight.size(2);
    int pad = K - 1;

    // Build padded input: (C_in, pad + T)
    int L_padded = pad + T;
    Tensor padded = Tensor::zeros({C_in, L_padded});

    if (initialized) {
        // Copy cache into the left part
        assert(cache.size(0) == C_in && cache.size(1) == pad);
        for (int c = 0; c < C_in; c++) {
            for (int i = 0; i < pad; i++) {
                padded(c, i) = cache(c, i);
            }
        }
    }
    // else: left part stays zero (zero-padding for first chunk)

    // Copy input into the right part
    for (int c = 0; c < C_in; c++) {
        for (int t = 0; t < T; t++) {
            padded(c, pad + t) = input(c, t);
        }
    }

    // Update cache: last (K-1) columns of padded
    if (!initialized) {
        cache = Tensor::zeros({C_in, pad});
        initialized = true;
    }
    for (int c = 0; c < C_in; c++) {
        for (int i = 0; i < pad; i++) {
            cache(c, i) = padded(c, L_padded - pad + i);
        }
    }

    // Run standard conv1d with padding=0 (now accelerated by BLAS when applicable)
    Tensor out = conv1d(padded, weight, bias, stride, 0, groups);
    return out;
}

Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  float eps) {
    // input: (T, D)
    int T = input.size(0);
    int D = input.size(1);
    Tensor out = Tensor::zeros({T, D});

    for (int t = 0; t < T; t++) {
        float mean = 0;
        const float* row = input.ptr() + t * D;
        for (int d = 0; d < D; d++) mean += row[d];
        mean /= D;

        float var = 0;
        for (int d = 0; d < D; d++) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        var /= D;

        float inv_std = 1.0f / std::sqrt(var + eps);
        float* out_row = out.ptr() + t * D;
        for (int d = 0; d < D; d++) {
            out_row[d] = (row[d] - mean) * inv_std * weight(d) + bias(d);
        }
    }
    return out;
}

Tensor layer_norm_1d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     float eps) {
    // input: (D) — single vector
    int D = input.size(0);
    Tensor out = Tensor::zeros({D});

    float mean = 0;
    for (int d = 0; d < D; d++) mean += input.data[d];
    mean /= D;

    float var = 0;
    for (int d = 0; d < D; d++) {
        float diff = input.data[d] - mean;
        var += diff * diff;
    }
    var /= D;

    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int d = 0; d < D; d++) {
        out.data[d] = (input.data[d] - mean) * inv_std * weight(d) + bias(d);
    }
    return out;
}

void relu_(Tensor& x) {
    for (auto& v : x.data)
        if (v < 0) v = 0;
}

void silu_(Tensor& x) {
    for (auto& v : x.data)
        v = v / (1.0f + std::exp(-v));
}

void softplus_(Tensor& x) {
    for (auto& v : x.data)
        v = std::log(1.0f + std::exp(v));
}

void leaky_relu_(Tensor& x, float slope) {
    for (auto& v : x.data)
        v = v >= 0 ? v : slope * v;
}

void tanh_(Tensor& x) {
    for (auto& v : x.data)
        v = std::tanh(v);
}

Tensor add(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    Tensor out;
    out.shape = a.shape;
    out.data.resize(a.numel());
    for (int i = 0; i < a.numel(); i++)
        out.data[i] = a.data[i] + b.data[i];
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    Tensor out;
    out.shape = a.shape;
    out.data.resize(a.numel());
    for (int i = 0; i < a.numel(); i++)
        out.data[i] = a.data[i] * b.data[i];
    return out;
}

} // namespace ops
