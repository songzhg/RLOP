#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include "torch_utils.h"

namespace py = pybind11;

namespace rlop::pybind11_utils {
    // Converts a Python tuple representing an array shape into a vector of integers suitable for
    // representing the sizes of a libtorch tensor.
    //
    // Parameters:
    //   shape: A py::tuple object representing the shape of a numpy array.
    //
    // Returns:
    //   std::vector<Int>: A vector containing the sizes of each dimension of the tensor.
    inline std::vector<Int> ArrayShapeToTensorSizes(const py::tuple& shape) {
        std::vector<Int> sizes;
        sizes.reserve(shape.size());
        for (const auto &it : shape) {
            sizes.push_back(py::cast<Int>(it));
        }
        return sizes;
    }

    // Converts a Python dtype to the corresponding libtorch Dtype.
    //
    // Parameters:
    //   dtype: A py::dtype object representing the data type of a numpy array.
    //
    // Returns:
    //   torch::Dtype: The corresponding libtorch data type.
    //
    // Throws:
    //   std::runtime_error if the numpy data type is unsupported.
    inline torch::Dtype ArrayDtypeToTensorDtype(const py::dtype& dtype) {
        if (dtype.equal(py::dtype::of<float>())) 
            return torch::kFloat32;
        else if (dtype.equal(py::dtype::of<double>()))
            return torch::kFloat64;
        else if (dtype.equal(py::dtype::of<int32_t>()))
            return torch::kInt32;
        else if (dtype.equal(py::dtype::of<int64_t>()))
            return torch::kInt64;
        else if (dtype.equal(py::dtype::of<int16_t>()))
            return torch::kInt16;
        else if (dtype.equal(py::dtype::of<int8_t>()))
            return torch::kInt8;
        else if (dtype.equal(py::dtype::of<uint8_t>()))
            return torch::kUInt8;
        else if (dtype.equal(py::dtype::of<bool>()))
            return torch::kBool;
        else 
            throw std::runtime_error("Unsupported data type");
    }

    //  std::string dtype_name = py::str(dtype.attr("name"));
    //         std::cout << dtype_name <<std::endl;
    //         if (dtype.is(py::dtype::of<float>())) 
    //             std::cout << "ss" <<std::endl;

    // Converts a libtorch Dtype to the corresponding Python dtype.
    //
    // Parameters:
    //   dtype: A torch::Dtype enum value representing the data type of a libtorch tensor.
    //
    // Returns:
    //   py::dtype: The corresponding Python dtype.
    //
    // Throws:
    //   std::runtime_error if the PyTorch data type is unsupported.
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

    // Converts a numpy array to a libtorch tensor, preserving the data type and shape.
    //
    // Parameters:
    //   array: A py::array object representing the numpy array to be converted.
    //
    // Returns:
    //   torch::Tensor: A libtorch tensor with the same data, data type, and shape as the numpy array.
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

    // Converts a libtorch tensor to a numpy array, ensuring memory continuity and correct data type representation.
    //
    // Parameters:
    //   tensor: A torch::Tensor object to be converted to a numpy array.
    //
    // Returns:
    //   py::array: A numpy array with the same data, data type, and shape as the PyTorch tensor.
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