#include "ops.h"
#include <cmath>
#include <cassert>
#include <cblas.h>
#include <cstring>

namespace ops {

Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    // input: (T, D_in), weight: (D_out, D_in), bias: (D_out)
    // output: (T, D_out) = input * weight^T + bias
    const int T = input.size(0);
    const int D_in = input.size(1);
    const int D_out = weight.size(0);
    assert(weight.size(1) == D_in);
    assert(bias.numel() == D_out);

    Tensor out = Tensor::zeros({T, D_out});

    // Fill bias first, then accumulate GEMM into it.
    for (int t = 0; t < T; ++t) {
        float* row = out.ptr() + t * D_out;
        std::memcpy(row, bias.ptr(), static_cast<size_t>(D_out) * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, D_out, D_in,
                1.0f,
                input.ptr(), D_in,
                weight.ptr(), D_in,
                1.0f,
                out.ptr(), D_out);
    return out;
}

Tensor linear_no_bias(const Tensor& input, const Tensor& weight) {
    // input: (T, D_in), weight: (D_out, D_in)
    // output: (T, D_out) = input * weight^T
    const int T = input.size(0);
    const int D_in = input.size(1);
    const int D_out = weight.size(0);
    assert(weight.size(1) == D_in);

    Tensor out = Tensor::zeros({T, D_out});
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, D_out, D_in,
                1.0f,
                input.ptr(), D_in,
                weight.ptr(), D_in,
                0.0f,
                out.ptr(), D_out);
    return out;
}

Tensor linear_prefix(const Tensor& input, int prefix_cols,
                     const Tensor& weight, const Tensor& bias) {
    // Uses input[:, :prefix_cols] without copying it into a separate Tensor.
    // Row-major GEMM supports a leading dimension larger than K.
    const int T = input.size(0);
    const int D_total = input.size(1);
    const int D_out = weight.size(0);
    assert(prefix_cols > 0 && prefix_cols <= D_total);
    assert(weight.size(1) == prefix_cols);
    assert(bias.numel() == D_out);

    Tensor out = Tensor::zeros({T, D_out});
    for (int t = 0; t < T; ++t) {
        float* row = out.ptr() + t * D_out;
        std::memcpy(row, bias.ptr(), static_cast<size_t>(D_out) * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, D_out, prefix_cols,
                1.0f,
                input.ptr(), D_total,      // lda keeps full row stride
                weight.ptr(), prefix_cols,
                1.0f,
                out.ptr(), D_out);
    return out;
}

Tensor linear_1d(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    const int D_in = input.size(0);
    const int D_out = weight.size(0);
    assert(weight.size(1) == D_in);
    assert(bias.numel() == D_out);

    Tensor out = Tensor::zeros({D_out});
    std::memcpy(out.ptr(), bias.ptr(), static_cast<size_t>(D_out) * sizeof(float));

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                D_out, D_in,
                1.0f,
                weight.ptr(), D_in,
                input.ptr(), 1,
                1.0f,
                out.ptr(), 1);
    return out;
}

Tensor linear_1d_no_bias(const Tensor& input, const Tensor& weight) {
    const int D_in = input.size(0);
    const int D_out = weight.size(0);
    assert(weight.size(1) == D_in);

    Tensor out = Tensor::zeros({D_out});
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                D_out, D_in,
                1.0f,
                weight.ptr(), D_in,
                input.ptr(), 1,
                0.0f,
                out.ptr(), 1);
    return out;
}

Tensor conv1d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride, int padding, int groups) {
    // input: (C_in, L)
    // weight: (C_out, C_in/groups, K)
    // output: (C_out, L_out)
<<<<<<< Updated upstream
    int C_in = input.size(0);
    int L = input.size(1);
    int C_out = weight.size(0);
    int K = weight.size(2);
    int C_in_per_group = C_in / groups;

    int L_out = (L + 2 * padding - K) / stride + 1;
=======
    const int C_in = input.size(0);
    const int L = input.size(1);
    const int C_out = weight.size(0);
    const int K = weight.size(2);
    const int C_in_per_group = C_in / groups;
    const int L_out = (L + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

>>>>>>> Stashed changes
    Tensor out = Tensor::zeros({C_out, L_out});
    const int channels_per_group_out = C_out / groups;

    // Fast path for all ordinary non-dilated convolutions.
    // The GEMM writes directly into the final (C_out, L_out) layout, avoiding
    // the old temporary (L_out, C_out) result and full transpose.
    if (groups == 1 && dilation == 1) {
        if (K == 1) {
            // Pointwise convolution: weight(C_out,C_in) * input(C_in,L_out).
            // For stride/padding that makes input columns non-contiguous, pack
            // only the sampled columns first.
            const float* b_ptr = nullptr;
            int b_ld = 0;
            std::vector<float> packed;

<<<<<<< Updated upstream
    // Use im2col + sgemm when groups==1 and K>1 (standard conv)
    // For depthwise (groups>1) or pointwise (K==1), BLAS overhead isn't worth it
    if (groups == 1 && K > 1) {
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
=======
            if (stride == 1 && padding == 0 && L_out == L) {
                b_ptr = input.ptr();
                b_ld = L;
            } else {
                packed.assign(static_cast<size_t>(C_in) * L_out, 0.0f);
                for (int ci = 0; ci < C_in; ++ci) {
                    float* dst = packed.data() + static_cast<size_t>(ci) * L_out;
                    const float* src = input.ptr() + static_cast<size_t>(ci) * L;
                    for (int ol = 0; ol < L_out; ++ol) {
                        const int il = ol * stride - padding;
                        if (il >= 0 && il < L) dst[ol] = src[il];
>>>>>>> Stashed changes
                    }
                }
                b_ptr = packed.data();
                b_ld = L_out;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        C_out, L_out, C_in,
                        1.0f,
                        weight.ptr(), C_in,
                        b_ptr, b_ld,
                        0.0f,
                        out.ptr(), L_out);
        } else {
            // im2col stored as (C_in*K, L_out), so GEMM directly produces
            // (C_out, L_out) with no post-GEMM transpose.
            const int Kdim = C_in * K;
            std::vector<float> col(static_cast<size_t>(Kdim) * L_out, 0.0f);

            for (int ci = 0; ci < C_in; ++ci) {
                const float* src = input.ptr() + static_cast<size_t>(ci) * L;
                for (int ki = 0; ki < K; ++ki) {
                    float* dst = col.data() + static_cast<size_t>(ci * K + ki) * L_out;

<<<<<<< Updated upstream
        // Transpose (L_out, C_out) -> (C_out, L_out) and add bias
        for (int co = 0; co < C_out; co++) {
            float b = bias(co);
            for (int ol = 0; ol < L_out; ol++) {
                out(co, ol) = out_T[ol * C_out + co] + b;
            }
        }
    } else {
        // Fallback: original loop-based implementation
        // Efficient for depthwise conv (groups == C_in) and pointwise (K==1)
        for (int g = 0; g < groups; g++) {
            int c_out_start = g * channels_per_group_out;
            int c_in_start = g * C_in_per_group;

            for (int co = c_out_start; co < c_out_start + channels_per_group_out; co++) {
                for (int ol = 0; ol < L_out; ol++) {
                    float sum = bias(co);
                    for (int ci = 0; ci < C_in_per_group; ci++) {
                        for (int ki = 0; ki < K; ki++) {
                            int il = ol * stride + ki - padding;
                            if (il >= 0 && il < L) {
                                sum += input(c_in_start + ci, il) *
                                       weight(co, ci, ki);
                            }
=======
                    if (padding == 0 && stride == 1) {
                        // Contiguous slice: input[ci, ki : ki+L_out]
                        std::memcpy(dst, src + ki,
                                    static_cast<size_t>(L_out) * sizeof(float));
                    } else if (padding == 0) {
                        for (int ol = 0; ol < L_out; ++ol) {
                            dst[ol] = src[ol * stride + ki];
                        }
                    } else {
                        for (int ol = 0; ol < L_out; ++ol) {
                            const int il = ol * stride + ki - padding;
                            if (il >= 0 && il < L) dst[ol] = src[il];
>>>>>>> Stashed changes
                        }
                    }
                }
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        C_out, L_out, Kdim,
                        1.0f,
                        weight.ptr(), Kdim,
                        col.data(), L_out,
                        0.0f,
                        out.ptr(), L_out);
        }

        // Bias is broadcast over time.
        for (int co = 0; co < C_out; ++co) {
            const float b = bias(co);
            float* row = out.ptr() + static_cast<size_t>(co) * L_out;
            for (int ol = 0; ol < L_out; ++ol) row[ol] += b;
        }
        return out;
    }

    // Generic path for dilated/depthwise/grouped convolutions.
    for (int g = 0; g < groups; ++g) {
        const int c_out_start = g * channels_per_group_out;
        const int c_in_start = g * C_in_per_group;

        for (int co = c_out_start; co < c_out_start + channels_per_group_out; ++co) {
            for (int ol = 0; ol < L_out; ++ol) {
                float sum = bias(co);
                for (int ci = 0; ci < C_in_per_group; ++ci) {
                    for (int ki = 0; ki < K; ++ki) {
                        const int il = ol * stride + ki * dilation - padding;
                        if (il >= 0 && il < L) {
                            sum += input(c_in_start + ci, il) * weight(co, ci, ki);
                        }
                    }
                }
                out(co, ol) = sum;
            }
        }
    }
    return out;
}

Tensor conv1d_causal(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int stride, Tensor& cache, bool& initialized, int groups) {
    const int C_in = input.size(0);
    const int T = input.size(1);
    const int K = weight.size(2);
    const int pad = K - 1;
    const int L_padded = pad + T;

    Tensor padded = Tensor::zeros({C_in, L_padded});

    // Copy by complete channel rows instead of scalar loops.
    for (int c = 0; c < C_in; ++c) {
        float* dst = padded.ptr() + static_cast<size_t>(c) * L_padded;
        if (initialized && pad > 0) {
            assert(cache.size(0) == C_in && cache.size(1) == pad);
            const float* cache_row = cache.ptr() + static_cast<size_t>(c) * pad;
            std::memcpy(dst, cache_row, static_cast<size_t>(pad) * sizeof(float));
        }

        const float* src = input.ptr() + static_cast<size_t>(c) * T;
        std::memcpy(dst + pad, src, static_cast<size_t>(T) * sizeof(float));
    }

    if (!initialized) {
        cache = Tensor::zeros({C_in, pad});
        initialized = true;
    }

    if (pad > 0) {
        for (int c = 0; c < C_in; ++c) {
            const float* src = padded.ptr() + static_cast<size_t>(c) * L_padded + (L_padded - pad);
            float* dst = cache.ptr() + static_cast<size_t>(c) * pad;
            std::memcpy(dst, src, static_cast<size_t>(pad) * sizeof(float));
        }
    }

    return conv1d(padded, weight, bias, stride, 0, groups);
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

Tensor add(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    Tensor out = a.clone();
    add_(out, b);
    return out;
}

void add_(Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    const int n = a.numel();
    for (int i = 0; i < n; ++i) a.data[i] += b.data[i];
}

Tensor mul(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    Tensor out = a.clone();
    mul_(out, b);
    return out;
}

void mul_(Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    const int n = a.numel();
    for (int i = 0; i < n; ++i) a.data[i] *= b.data[i];
}

} // namespace ops
