#pragma once
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>
#include <iostream>

struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() = default;

    int dim() const { return (int)shape.size(); }

    int size(int d) const {
        if (d < 0) d += dim();
        return shape[d];
    }

    int numel() const {
        if (shape.empty()) return 0;
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }

    int stride(int d) const {
        int s = 1;
        for (int i = d + 1; i < dim(); i++) s *= shape[i];
        return s;
    }

    float& operator()(int i) { return data[i]; }
    float  operator()(int i) const { return data[i]; }

    float& operator()(int i, int j) { return data[i * shape[1] + j]; }
    float  operator()(int i, int j) const { return data[i * shape[1] + j]; }

    float& operator()(int i, int j, int k) {
        return data[i * shape[1] * shape[2] + j * shape[2] + k];
    }
    float operator()(int i, int j, int k) const {
        return data[i * shape[1] * shape[2] + j * shape[2] + k];
    }

    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }

    static Tensor zeros(std::vector<int> sh) {
        Tensor t;
        t.shape = sh;
        int n = 1;
        for (int s : sh) n *= s;
        t.data.assign(n, 0.0f);
        return t;
    }

    static Tensor from_data(const float* src, std::vector<int> sh) {
        Tensor t;
        t.shape = sh;
        int n = 1;
        for (int s : sh) n *= s;
        t.data.assign(src, src + n);
        return t;
    }

    void fill(float val) { std::fill(data.begin(), data.end(), val); }

    Tensor clone() const {
        Tensor t;
        t.shape = shape;
        t.data = data;
        return t;
    }

    Tensor reshape(std::vector<int> new_shape) const {
        Tensor t;
        t.data = data;
        t.shape = new_shape;
        return t;
    }

    Tensor transpose_2d() const {
        assert(dim() == 2);
        int r = shape[0], c = shape[1];
        Tensor t = Tensor::zeros({c, r});
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                t(j, i) = (*this)(i, j);
        return t;
    }

    // Transpose (C, L) <-> (L, C) for convenience
    Tensor transpose_12() const {
        assert(dim() == 3);
        int d0 = shape[0], d1 = shape[1], d2 = shape[2];
        Tensor t = Tensor::zeros({d0, d2, d1});
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    t(i, k, j) = (*this)(i, j, k);
        return t;
    }

    void print_shape(const std::string& name = "") const {
        if (!name.empty()) std::cerr << name << ": ";
        std::cerr << "(";
        for (int i = 0; i < dim(); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << shape[i];
        }
        std::cerr << ")" << std::endl;
    }
};
