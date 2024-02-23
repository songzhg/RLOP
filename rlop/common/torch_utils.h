#pragma once
#include "torch/torch.h"
#include "typedef.h"

namespace rlop::torch_utils {
    // Copies the state dictionary from a source libtorch model to a target model. This includes both
    // the parameters and buffers (e.g., running mean/variance in batch normalization layers).
    //
    // Parameters:
    //   source_model: The model from which to copy parameters and buffers.
    //   target_model: Pointer to the model where the parameters and buffers will be copied to.
    inline void CopyStateDict(const torch::nn::Module& source_model, torch::nn::Module* target_model) {
        torch::NoGradGuard no_grad;
        auto params = source_model.named_parameters(true /*recurse*/);
        auto buffers = source_model.named_buffers(true /*recurse*/);
        auto target_params = target_model->named_parameters(true /*recurse*/);
        auto target_buffers = target_model->named_buffers(true /*recurse*/);
        for (auto& val : params) {
            auto name = val.key();
            if (target_params.contains(name)) {
                auto* t_param = target_params.find(name);
                t_param->copy_(val.value());
            }
        }
        for (auto& val : buffers) {
            auto name = val.key();
            if (target_buffers.contains(name)) {
                auto* t_buffer = target_buffers.find(name);
                t_buffer->copy_(val.value());
            }
        }
    }

    // Retrieves the names and tensors of all parameters in a libtorch model.
    //
    // Parameters:
    //   model: The model from which to retrieve parameters.
    //
    // Returns:
    //   A pair consisting of a vector of parameter names and a vector of parameter tensors.
    inline std::pair<std::vector<std::string>, std::vector<torch::Tensor>> GetParameters(const torch::nn::Module& model) {
        std::vector<std::string> names;
        std::vector<torch::Tensor> parameters;
        for (const auto& param : model.named_parameters()) {
            names.push_back(param.key());
            parameters.push_back(param.value());
        }
        return { names, parameters };
    }

    // Retrieves the names and tensors of all buffers in a libtorch model.
    //
    // Parameters:
    //   model: The model from which to retrieve buffers.
    //
    // Returns:
    //   A pair consisting of a vector of buffer names and a vector of buffer tensors.
    inline std::pair<std::vector<std::string>, std::vector<torch::Tensor>> GetBuffers(const torch::nn::Module& model) {
        std::vector<std::string> names;
        std::vector<torch::Tensor> buffers;
        for (const auto& buffer : model.named_buffers()) {
            names.push_back(buffer.key());
            buffers.push_back(buffer.value());
        }
        return { names, buffers };
    }

    // Applies Polyak averaging to update target parameters towards source parameters.
    //
    // Parameters:
    //   params: Source parameters to update from.
    //   target_params: Target parameters to update.
    //   tau: The interpolation parameter controlling the update extent.
    inline void PolyakUpdate(const std::vector<torch::Tensor>& params, std::vector<torch::Tensor>& target_params, double tau) {
        tau = std::clamp(tau, 0.0, 1.0);
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < params.size(); ++i) {
            if (i >= target_params.size()) {
                throw std::runtime_error("PolyakUpdate: mismatch in the number of parameters and target parameters");
            }
            target_params[i].data().mul_(1 - tau).add_(params[i].data(), tau);
        }
    }

    // Compares two libtorch models to check if they are identical in terms of parameters and buffers.
    //
    // Parameters:
    //   model1: The first model to compare.
    //   model2: The second model to compare.
    //
    // Returns:
    //   bool: True if both models have identical parameters and buffers, false otherwise.
    inline bool CompareModels(const torch::nn::Module& model1, const torch::nn::Module& model2) {
        auto params1 = model1.named_parameters();
        auto params2 = model2.named_parameters();
        auto buffers1 = model1.named_buffers();
        auto buffers2 = model2.named_buffers();
        
        if (params1.size() != params2.size()) 
            return false;
        if (buffers1.size() != buffers2.size())
            return false;
        for (size_t i = 0; i < params1.size(); ++i) {
            const auto& kvp1 = params1[i];
            const auto& kvp2 = params2[i];
            if (kvp1.key() != kvp2.key())
                return false;
            if (!torch::equal(kvp1.value(), kvp2.value()))
                return false;
        }
        for (size_t i = 0; i < buffers1.size(); ++i) {
            const auto& kvp1 = buffers1[i];
            const auto& kvp2 = buffers2[i];
            if (kvp1.key() != kvp2.key())
                return false;
            if (!torch::equal(kvp1.value(), kvp2.value()))
                return false;
        }
        return true;
    }

    // Computes the approximate Kullback–Leibler divergence between two distributions given their log probabilities.
    //
    // Parameters:
    //   log_prob1: Log probabilities of the first distribution.
    //   log_prob2: Log probabilities of the second distribution.
    //
    // Returns:
    //   torch::Tensor: The computed approximate KL divergence.
    inline torch::Tensor ComputeApproxKL(const torch::Tensor& log_prob1, const torch::Tensor& log_prob2) {
        torch::Tensor log_ratio = log_prob1 - log_prob2;
        return torch::mean((torch::exp(log_ratio) - 1) - log_ratio);
    }

    // Computes the total byte size required to store tensors of given sizes and data type.
    //
    // Parameters:
    //   sizes: Sizes of the tensors.
    //   type: Data type of the tensors.
    //
    // Returns:
    //   size_t: The total byte size required.
    inline size_t ComputeByteSize(const std::vector<Int>& sizes, const torch::Dtype& type) {
        Int num_elements = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<Int>());
        size_t element_size = torch::elementSize(type);
        return num_elements * element_size;
    }

    // Computes the explained variance, a measure of how well a model's predictions approximate the true values.
    //
    // Parameters:
    //   y_pred: Predicted values.
    //   y_true: True values.
    //
    // Returns:
    //   torch::Tensor: The explained variance.
    inline torch::Tensor ExplainedVariance(const torch::Tensor& y_pred, const torch::Tensor& y_true) {
        TORCH_CHECK(y_true.dim() == 1 && y_pred.dim() == 1, "y_true and y_pred must be 1-dimensional");
        auto var_y = torch::var(y_true, /* unbiased */ false);
        if (var_y.item<double>() == 0) {
            return torch::tensor(NAN);
        }
        auto ev = 1 - torch::var(y_true - y_pred, /* unbiased */ false) / var_y;
        return ev;
    }

    // Computes the hyperbolic arctangent of a tensor, with clamping to avoid numerical errors.
    //
    // Parameters:
    //   value: The input tensor.
    //
    // Returns:
    //   torch::Tensor: The hyperbolic
    inline torch::Tensor Atanh(const torch::Tensor& value) {
        auto eps = std::numeric_limits<decltype(value.item().toFloat())>::epsilon();
        torch::Tensor clamped_value = torch::clamp(value, -1.0 + eps, 1.0 - eps);
        return torch::atanh(clamped_value);
    }
}