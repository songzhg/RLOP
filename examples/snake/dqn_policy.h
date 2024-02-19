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

        void Reset() override {
            for (auto& module : children()) {
                if (auto conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
                    torch::nn::init::xavier_uniform_(conv->weight);
                    torch::nn::init::constant_(conv->bias, 0);
                }
                else if (auto linear = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
                    torch::nn::init::xavier_uniform_(linear->weight);
                    torch::nn::init::constant_(linear->bias, 0);
                }
            }
        }

        torch::Tensor Forward(const torch::Tensor& observation) override {
            torch::Tensor feature = feature_extractor_->forward(observation);
            return mlp_->forward(feature);
        }

    private:
        torch::nn::Sequential feature_extractor_;
        torch::nn::Sequential mlp_;
    };
}