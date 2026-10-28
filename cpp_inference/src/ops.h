#pragma once
#include "tensor.h"

namespace ops {

// Linear: input(T, D_in) x weight(D_out, D_in)^T + bias(D_out) -> (T, D_out)
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

// Linear for single vector: input(D_in) x weight(D_out, D_in)^T + bias(D_out) -> (D_out)
Tensor linear_1d(const Tensor& input, const Tensor& weight, const Tensor& bias);

// Conv1d: input(C_in, L), weight(C_out, C_in, K) -> (C_out, L_out)
Tensor conv1d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride = 1, int padding = 0, int groups = 1, int dilation = 1);

// ConvTranspose1d: input(C_in, L), weight(C_in, C_out, K) -> (C_out, L_out)
// L_out = (L - 1) * stride - 2 * padding + K
Tensor conv_transpose1d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                        int stride = 1, int padding = 0);

// Causal Conv1d with cache for streaming:
//   input:  (C_in, T)
//   weight: (C_out, C_in/groups, K)
//   cache:  (C_in, K-1)  — left-context from previous chunk
//   output: (C_out, T_out)
// The cache is updated in-place with the last K-1 columns of the padded input.
// On first call, set initialized=false; cache will be zero-padded.
Tensor conv1d_causal(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     int stride, Tensor& cache, bool& initialized, int groups = 1);

// LayerNorm: input(T, D), weight(D), bias(D) -> (T, D)
Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  float eps = 1e-5f);

// LayerNorm for single vector: input(D), weight(D), bias(D) -> (D)
Tensor layer_norm_1d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     float eps = 1e-5f);

// Activation functions (in-place)
void relu_(Tensor& x);
void silu_(Tensor& x);      // x * sigmoid(x)
void softplus_(Tensor& x);  // log(1 + exp(x))
void leaky_relu_(Tensor& x, float slope = 0.1f);
void tanh_(Tensor& x);

// Element-wise: out = a + b (same shape)
Tensor add(const Tensor& a, const Tensor& b);

// Element-wise multiply
Tensor mul(const Tensor& a, const Tensor& b);

} // namespace ops
