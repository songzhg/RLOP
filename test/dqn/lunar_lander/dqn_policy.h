#pragma once
#include "rlop/rl/dqn/policy.h"

namespace lunar_lander {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class QNet : public rlop::QNet {
    public:
        QNet(Int observation_dim, Int num_actions) {
            mlp_->push_back(torch::nn::Linear(observation_dim, 64));
            mlp_->push_back(torch::nn::ReLU());
            mlp_->push_back(torch::nn::Linear(64, 64));
            mlp_->push_back(torch::nn::ReLU());
            mlp_->push_back(torch::nn::Linear(64, num_actions));
            register_module("mlp", mlp_);
        }

        torch::Tensor PredictQValues(const torch::Tensor& observations) override {
            return mlp_->forward(observations);
        }

    private:
        torch::nn::Sequential mlp_;
    };

    class DQNPolicy : public rlop::DQNPolicy {
    public:
        DQNPolicy(Int observation_dim, Int num_actions) :
            observation_dim_(observation_dim),
            num_actions_(num_actions)
        {}

        std::shared_ptr<rlop::QNet> MakeQNet() const override {
            return std::make_shared<QNet>(observation_dim_, num_actions_);
        }

    private:
        Int observation_dim_;
        Int num_actions_;
    };
}