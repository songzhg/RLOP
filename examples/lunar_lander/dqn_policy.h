#pragma once
#include "rlop/rl/dqn/policy.h"

namespace lunar_lander {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class QNet : public rlop::QNet {
    public:
        QNet(Int observation_dim, Int num_actions) :
            mlp_(
                torch::nn::Linear(observation_dim, 64),
                torch::nn::ReLU(),
                torch::nn::Linear(64, 64),
                torch::nn::ReLU(),
                torch::nn::Linear(64, num_actions)
            )
        {
            register_module("mlp", mlp_);
        }

        void Reset() override {}

        torch::Tensor Forward(const torch::Tensor& observations) override {
            return mlp_->forward(observations);
        }

    private:
        torch::nn::Sequential mlp_;
    };
}