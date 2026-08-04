import base64, os, json

files = {}

files["cpp_inference/CMakeLists.txt"] = """cmake_minimum_required(VERSION 3.10)
project(emg_inference CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")

add_executable(emg_inference
    src/main.cpp
    src/ops.cpp
    src/emg_model.cpp
)
target_include_directories(emg_inference PRIVATE src)
"""

files["cpp_inference/src/tensor.h"] = r"""#pragma once
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
    int size(int d) const { if (d < 0) d += dim(); return shape[d]; }
    int numel() const { if (shape.empty()) return 0; int n=1; for(int s:shape) n*=s; return n; }
    int stride(int d) const { int s=1; for(int i=d+1;i<dim();i++) s*=shape[i]; return s; }
    float& operator()(int i) { return data[i]; }
    float  operator()(int i) const { return data[i]; }
    float& operator()(int i, int j) { return data[i*shape[1]+j]; }
    float  operator()(int i, int j) const { return data[i*shape[1]+j]; }
    float& operator()(int i, int j, int k) { return data[i*shape[1]*shape[2]+j*shape[2]+k]; }
    float  operator()(int i, int j, int k) const { return data[i*shape[1]*shape[2]+j*shape[2]+k]; }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
    static Tensor zeros(std::vector<int> sh) { Tensor t; t.shape=sh; int n=1; for(int s:sh) n*=s; t.data.assign(n,0.0f); return t; }
    static Tensor from_data(const float* src, std::vector<int> sh) { Tensor t; t.shape=sh; int n=1; for(int s:sh) n*=s; t.data.assign(src,src+n); return t; }
    void fill(float val) { std::fill(data.begin(),data.end(),val); }
    Tensor clone() const { Tensor t; t.shape=shape; t.data=data; return t; }
    Tensor reshape(std::vector<int> ns) const { Tensor t; t.data=data; t.shape=ns; return t; }
    Tensor transpose_2d() const { assert(dim()==2); int r=shape[0],c=shape[1]; Tensor t=zeros({c,r}); for(int i=0;i<r;i++) for(int j=0;j<c;j++) t(j,i)=(*this)(i,j); return t; }
    void print_shape(const std::string& name="") const { if(!name.empty()) std::cerr<<name<<": "; std::cerr<<"("; for(int i=0;i<dim();i++){if(i>0)std::cerr<<", ";std::cerr<<shape[i];} std::cerr<<")"<<std::endl; }
};
"""

# Encode and write all to a single python-generated bash script
base = "/home/maxch/mambassm"
for path, content in files.items():
    full = os.path.join(base, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(content)
    print(f"Wrote {full} ({len(content)} bytes)")