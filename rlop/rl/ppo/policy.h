#pragma once
#include "rlop/rl/policy.h"

namespace rlop {
    class PPOPolicy : public RLPolicy {
    public:
        PPOPolicy() = default;

        virtual ~PPOPolicy() = default;

        virtual torch::Tensor PredictValue(const torch::Tensor& observation) = 0;

        virtual std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateAction(const torch::Tensor& observation, const torch::Tensor& action) = 0;

        virtual std::array<torch::Tensor, 3> Forward(const torch::Tensor& observation) = 0;
    };
}