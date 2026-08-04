// pybind11 binding for C++ inference engine.
// Exposes CppInferenceEngine with clean list[float] interface -- no torch/numpy.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "emg_model.h"
#include "weight_loader.h"

namespace py = pybind11;

class CppInferenceEngine {
    EmgModel model;
    WeightMap wm;
    StreamingState state;
    bool loaded = false;
    int num_outs = 0;

public:
    void load_weights(const std::string& dir) {
        wm.load_manifest(dir);
        model.load(wm);
        num_outs = model.get_num_outs();
        loaded = true;
    }

    void init_state() {
        if (!loaded) {
            throw std::runtime_error("Must call load_weights() before init_state()");
        }
        state = model.init_streaming_state();
    }

    /// forward_streaming: takes flat list[float] of size (chunk_len * 8),
    /// returns list[list[float]] of shape (chunk_frames, 80).
    py::list forward_streaming(py::list emg_list, int chunk_len) {
        if (!loaded) {
            throw std::runtime_error("Must call load_weights() before forward_streaming()");
        }

        int num_channels = 8;
        int total = chunk_len * num_channels;

        // Convert py::list -> Tensor (chunk_len, 8)
        std::vector<float> emg_data(total);
        for (int i = 0; i < total; i++) {
            emg_data[i] = emg_list[i].cast<float>();
        }
        Tensor input = Tensor::from_data(emg_data.data(), {chunk_len, num_channels});

        // Run streaming forward
        Tensor mel = model.forward_streaming(input, state);

        // Convert Tensor (frames, num_outs) -> list[list[float]]
        int frames = mel.size(0);
        int outs = mel.size(1);
        py::list result;
        for (int f = 0; f < frames; f++) {
            py::list row;
            for (int o = 0; o < outs; o++) {
                row.append(mel(f, o));
            }
            result.append(row);
        }
        return result;
    }

    /// Batch forward (for testing): takes flat list[float] of size (T * 8),
    /// returns list[list[float]] of shape (T/8, 80).
    py::list forward_batch(py::list emg_list, int T) {
        if (!loaded) {
            throw std::runtime_error("Must call load_weights() before forward_batch()");
        }

        int num_channels = 8;
        int total = T * num_channels;

        std::vector<float> emg_data(total);
        for (int i = 0; i < total; i++) {
            emg_data[i] = emg_list[i].cast<float>();
        }
        Tensor input = Tensor::from_data(emg_data.data(), {T, num_channels});

        Tensor mel = model.forward(input);

        int frames = mel.size(0);
        int outs = mel.size(1);
        py::list result;
        for (int f = 0; f < frames; f++) {
            py::list row;
            for (int o = 0; o < outs; o++) {
                row.append(mel(f, o));
            }
            result.append(row);
        }
        return result;
    }
};

PYBIND11_MODULE(cpp_forward, m) {
    m.doc() = "C++ EMG-to-Mel inference engine (no torch/numpy dependency)";

    py::class_<CppInferenceEngine>(m, "CppInferenceEngine")
        .def(py::init<>())
        .def("load_weights", &CppInferenceEngine::load_weights,
             py::arg("weights_dir"),
             "Load model weights from exported_weights directory")
        .def("init_state", &CppInferenceEngine::init_state,
             "Initialize streaming state (call before first forward_streaming)")
        .def("forward_streaming", &CppInferenceEngine::forward_streaming,
             py::arg("emg"), py::arg("chunk_len"),
             "Streaming forward: list[float](chunk_len*8) -> list[list[float]](frames, 80)")
        .def("forward_batch", &CppInferenceEngine::forward_batch,
             py::arg("emg"), py::arg("T"),
             "Batch forward: list[float](T*8) -> list[list[float]](T/8, 80)");
}
