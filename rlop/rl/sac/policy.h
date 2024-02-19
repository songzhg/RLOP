#pragma once
#include "rlop/rl/policy.h"

namespace rlop {
    class SACActor : public RLPolicy {
    public:
        SACActor() = default;

        virtual ~SACActor() = default;

        virtual std::array<torch::Tensor, 2> PredictActionLogProb(const torch::Tensor& observation) = 0;
    };

    class SACCritic : public torch::nn::Module {
    public:
        virtual ~SACCritic() = default;

        virtual void Reset() = 0;

        virtual std::vector<torch::Tensor> Forward(const torch::Tensor& observation, const torch::Tensor& action) = 0;
    };
}