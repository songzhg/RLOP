#pragma once
#include "rlop/common/torch_utils.h"

namespace rlop {
    class RLPolicy : public torch::nn::Module {
    public:
        RLPolicy() = default;

        virtual ~RLPolicy() = default;

        virtual void Reset() = 0;

        virtual torch::Tensor PredictAction(const torch::Tensor& observation, bool deterministic = false) = 0;

        virtual std::array<torch::Tensor, 2> Predict(const torch::Tensor& observation, bool deterministic = false, const torch::Tensor& state = torch::Tensor(), const torch::Tensor& episode_start = torch::Tensor()) {
            this->eval();
            torch::NoGradGuard no_grad;
            return { PredictAction(observation, deterministic), torch::Tensor() };
        }
    };
}