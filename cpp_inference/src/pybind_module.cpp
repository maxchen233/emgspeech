// pybind_module_pruned.cpp
//
// pybind11 binding for the pruned CNN + Mamba C++ engine.
//
// Two interfaces are kept:
//
// 1) Fast NumPy interface (recommended)
//      mel = engine.forward_streaming(emg)
//      emg: np.float32 [T, 8]
//      mel: np.float32 [frames, num_outs]
//
// 2) Old list interface (backward compatible)
//      mel = engine.forward_streaming(flat_list, chunk_len)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

#include "emg_model.h"
#include "weight_loader.h"

namespace py = pybind11;

class CppInferenceEngine {
    EmgModel model;
    StreamingState state;
    bool loaded = false;
    int num_outs = 0;

public:
    void load_weights(const std::string& dir) {
        // Keep the loader temporary. model.load() owns its Tensor copies, so
        // retaining WeightMap::cache would otherwise keep a second full copy
        // of the model weights in Raspberry Pi RAM.
        WeightMap wm;
        wm.load_manifest(dir);
        model.load(wm);

        num_outs = model.get_num_outs();
        loaded = true;

        std::cerr << "CNN mid channels: ["
                  << model.get_cnn_mid_channels(0) << ", "
                  << model.get_cnn_mid_channels(1) << ", "
                  << model.get_cnn_mid_channels(2) << "]"
                  << std::endl;
    }

    void init_state() {
        if (!loaded) {
            throw std::runtime_error(
                "Must call load_weights() before init_state()"
            );
        }

        state = model.init_streaming_state();
    }

    // ---------------------------------------------------------
    // Recommended NumPy streaming interface
    // ---------------------------------------------------------
    py::array_t<float> forward_streaming_numpy(
        py::array_t<float,
            py::array::c_style |
            py::array::forcecast> emg
    ) {
        if (!loaded) {
            throw std::runtime_error(
                "Must call load_weights() before forward_streaming()"
            );
        }

        auto buf = emg.request();

        if (buf.ndim != 2) {
            throw std::runtime_error(
                "EMG must be a 2-D float32 array with shape (T, 8)"
            );
        }

        if (buf.shape[1] != 8) {
            throw std::runtime_error(
                "EMG second dimension must be 8"
            );
        }

        const int T = static_cast<int>(buf.shape[0]);
        const float* ptr = static_cast<const float*>(buf.ptr);

        Tensor input = Tensor::from_data(
            ptr,
            {T, 8}
        );

        Tensor mel;

        {
            // C++ inference itself does not need the Python GIL.
            py::gil_scoped_release release;
            mel = model.forward_streaming(
                input,
                state
            );
        }

        const int frames = mel.size(0);
        const int outs = mel.size(1);

        py::array_t<float> result(
            {frames, outs}
        );

        std::memcpy(
            result.mutable_data(),
            mel.ptr(),
            static_cast<size_t>(frames) * outs * sizeof(float)
        );

        return result;
    }

    // ---------------------------------------------------------
    // Recommended NumPy batch interface
    // ---------------------------------------------------------
    py::array_t<float> forward_batch_numpy(
        py::array_t<float,
            py::array::c_style |
            py::array::forcecast> emg
    ) {
        if (!loaded) {
            throw std::runtime_error(
                "Must call load_weights() before forward_batch()"
            );
        }

        auto buf = emg.request();

        if (buf.ndim != 2 || buf.shape[1] != 8) {
            throw std::runtime_error(
                "EMG must have shape (T, 8)"
            );
        }

        const int T = static_cast<int>(buf.shape[0]);
        const float* ptr = static_cast<const float*>(buf.ptr);

        Tensor input = Tensor::from_data(
            ptr,
            {T, 8}
        );

        Tensor mel;

        {
            py::gil_scoped_release release;
            mel = model.forward(input);
        }

        const int frames = mel.size(0);
        const int outs = mel.size(1);

        py::array_t<float> result(
            {frames, outs}
        );

        std::memcpy(
            result.mutable_data(),
            mel.ptr(),
            static_cast<size_t>(frames) * outs * sizeof(float)
        );

        return result;
    }

    // ---------------------------------------------------------
    // Legacy list interface
    // ---------------------------------------------------------
    py::list forward_streaming_list(
        py::list emg_list,
        int chunk_len
    ) {
        if (!loaded) {
            throw std::runtime_error(
                "Must call load_weights() before forward_streaming()"
            );
        }

        const int total = chunk_len * 8;

        if (static_cast<int>(py::len(emg_list)) != total) {
            throw std::runtime_error(
                "flat EMG list size != chunk_len * 8"
            );
        }

        std::vector<float> emg_data(total);

        for (int i = 0; i < total; i++) {
            emg_data[i] = emg_list[i].cast<float>();
        }

        Tensor input = Tensor::from_data(
            emg_data.data(),
            {chunk_len, 8}
        );

        Tensor mel = model.forward_streaming(
            input,
            state
        );

        const int frames = mel.size(0);
        const int outs = mel.size(1);

        py::list result;

        for (int f = 0; f < frames; f++) {
            py::list row;

            for (int o = 0; o < outs; o++) {
                row.append(
                    mel(f, o)
                );
            }

            result.append(row);
        }

        return result;
    }

    py::list forward_batch_list(
        py::list emg_list,
        int T
    ) {
        if (!loaded) {
            throw std::runtime_error(
                "Must call load_weights() before forward_batch()"
            );
        }

        const int total = T * 8;

        if (static_cast<int>(py::len(emg_list)) != total) {
            throw std::runtime_error(
                "flat EMG list size != T * 8"
            );
        }

        std::vector<float> emg_data(total);

        for (int i = 0; i < total; i++) {
            emg_data[i] = emg_list[i].cast<float>();
        }

        Tensor input = Tensor::from_data(
            emg_data.data(),
            {T, 8}
        );

        Tensor mel = model.forward(input);

        const int frames = mel.size(0);
        const int outs = mel.size(1);

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

    int get_num_outs() const {
        return num_outs;
    }

    py::tuple get_cnn_mid_channels() const {
        return py::make_tuple(
            model.get_cnn_mid_channels(0),
            model.get_cnn_mid_channels(1),
            model.get_cnn_mid_channels(2)
        );
    }
};


PYBIND11_MODULE(cpp_forward, m) {
    m.doc() =
        "C++ pruned CNN-Mamba inference engine";

    py::class_<CppInferenceEngine>(
        m,
        "CppInferenceEngine"
    )
        .def(py::init<>())

        .def(
            "load_weights",
            &CppInferenceEngine::load_weights,
            py::arg("weights_dir")
        )

        .def(
            "init_state",
            &CppInferenceEngine::init_state
        )

        // NumPy overload: engine.forward_streaming(emg_np)
        .def(
            "forward_streaming",
            &CppInferenceEngine::forward_streaming_numpy,
            py::arg("emg")
        )

        // Legacy overload:
        // engine.forward_streaming(flat_list, chunk_len)
        .def(
            "forward_streaming",
            &CppInferenceEngine::forward_streaming_list,
            py::arg("emg"),
            py::arg("chunk_len")
        )

        .def(
            "forward_batch",
            &CppInferenceEngine::forward_batch_numpy,
            py::arg("emg")
        )

        .def(
            "forward_batch",
            &CppInferenceEngine::forward_batch_list,
            py::arg("emg"),
            py::arg("T")
        )

        .def(
            "get_num_outs",
            &CppInferenceEngine::get_num_outs
        )

        .def(
            "get_cnn_mid_channels",
            &CppInferenceEngine::get_cnn_mid_channels
        );
}
