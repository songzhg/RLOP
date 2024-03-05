#pragma once
#include "rlop/rl/dqn/policy.h"

namespace snake {
    class QNet : public rlop::QNet {
    public:
        QNet(const std::vector<Int>& observation_sizes, Int num_actions) :
            feature_extractor_(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(observation_sizes[0], 16, 3).padding({1, 1})),
                torch::nn::ReLU(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3)),
                torch::nn::ReLU(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)),
                torch::nn::ReLU(),
                torch::nn::Flatten()
            ),
            mlp_(
                torch::nn::Linear(64 * (observation_sizes[1] - 4) * (observation_sizes[2] - 4), 256),
                torch::nn::ReLU(),
                torch::nn::Linear(256, num_actions)
            )
        {
            register_module("feature_extractor", feature_extractor_);
            register_module("mlp", mlp_);
        }

        void Reset() override {}

        torch::Tensor Forward(const torch::Tensor& observations) override {
            torch::Tensor features = feature_extractor_->forward(observations);
            return mlp_->forward(features);
        }

    private:
        torch::nn::Sequential feature_extractor_;
        torch::nn::Sequential mlp_;
    };
}