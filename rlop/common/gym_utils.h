#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include "torch_utils.h"

namespace py = pybind11;

namespace rlop::gym_utils {
    inline std::vector<Int> ArrayShapeToTensorSizes(const py::tuple& shape) {
        std::vector<Int> sizes;
        sizes.reserve(shape.size());
        for (const auto &it : shape) {
            sizes.push_back(py::cast<Int>(it));
        }
        return sizes;
    }

    inline torch::Dtype ArrayDtypeToTensorDtype(const py::dtype& dtype) {
        if (dtype.is(py::dtype::of<float>())) 
            return torch::kFloat32;
        else if (dtype.is(py::dtype::of<double>()))
            return torch::kFloat64;
        else if (dtype.is(py::dtype::of<int32_t>()))
            return torch::kInt32;
        else if (dtype.is(py::dtype::of<int64_t>()))
            return torch::kInt64;
        else if (dtype.is(py::dtype::of<int16_t>()))
            return torch::kInt16;
        else if (dtype.is(py::dtype::of<int8_t>()))
            return torch::kInt8;
        else if (dtype.is(py::dtype::of<uint8_t>()))
            return torch::kUInt8;
        else if (dtype.is(py::dtype::of<bool>()))
            return torch::kBool;
        else
            throw std::runtime_error("Unsupported data type");
    }

    inline py::dtype TensorDtypeToArrayDtype(const torch::Dtype& dtype) {
        switch (dtype) {
        case torch::kFloat32:
            return py::dtype("float32");
        case torch::kFloat64:
            return py::dtype("float64");
        case torch::kInt32:
            return py::dtype("int32");
        case torch::kInt64:
            return py::dtype("int64");
        case torch::kInt16:
            return py::dtype("int16");
        case torch::kInt8:
            return py::dtype("int8");
        case torch::kUInt8:
            return py::dtype("uint8");
        case torch::kBool:
            return py::dtype("bool");
        default:
            throw std::runtime_error("Unsupported data type");
        }
    }

    inline torch::Tensor ArrayToTensor(const py::array& array) {
        py::buffer_info info = array.request();
        torch::TensorOptions options = torch::TensorOptions().dtype(ArrayDtypeToTensorDtype(array.dtype()));
        std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
        std::vector<int64_t> stride(info.strides.begin(), info.strides.end());
        for (auto& s : stride) {
            s /= info.itemsize;
        }
        return torch::from_blob(info.ptr, shape, stride, options).clone();
    }

    inline py::array TensorToArray(const torch::Tensor& tensor) {
        torch::Tensor tensor_tmp = tensor.contiguous().cpu();
        pybind11::dtype dtype = TensorDtypeToArrayDtype(tensor_tmp.scalar_type());
        std::vector<ssize_t> shape = tensor_tmp.sizes().vec();
        std::vector<ssize_t> strides = tensor_tmp.strides().vec();
        for (auto& stride : strides) {
            stride *= tensor_tmp.element_size();
        }
        return pybind11::array(dtype, shape, strides, tensor_tmp.data_ptr());
    }
}