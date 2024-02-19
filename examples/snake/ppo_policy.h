#pragma once
#include "rlop/rl/ppo/policy.h"

namespace snake {
    class PPOPolicy : public rlop::PPOPolicy {
    public:
        PPOPolicy(const std::vector<Int>& observation_sizes, Int num_actions) :
            feature_extractor_(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(observation_sizes[0], 16, 3).padding({1, 1})),
                torch::nn::ReLU(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3)),
                torch::nn::ReLU(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)),
                torch::nn::ReLU(),
                torch::nn::Flatten()
            ),
            action_mlp_(64 * (observation_sizes[1] - 4) * (observation_sizes[2] - 4), 256),
            value_mlp_(64 * (observation_sizes[1] - 4) * (observation_sizes[2] - 4), 64),
            action_net_(256, num_actions),
            value_net_(64, 1)
        {
            register_module("feature_extractor", feature_extractor_);
            register_module("action_mlp", action_mlp_);
            register_module("value_mlp", value_mlp_);
            register_module("action_net", action_net_);
            register_module("value_net", value_net_);
        }

        void Reset() override {
            for (auto& module : feature_extractor_->modules()) {
                if (auto conv = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
                    torch::nn::init::orthogonal_(conv->weight, std::sqrt(2));
                    torch::nn::init::constant_(conv->bias, 0);
                }
            }
            torch::nn::init::orthogonal_(action_mlp_->weight, std::sqrt(2));
            torch::nn::init::constant_(action_mlp_->bias, 0);
            torch::nn::init::orthogonal_(value_mlp_->weight, std::sqrt(2));
            torch::nn::init::constant_(value_mlp_->bias, 0);
            torch::nn::init::orthogonal_(action_net_->weight, 0.01);
            torch::nn::init::constant_(action_net_->bias, 0);
            torch::nn::init::orthogonal_(value_net_->weight, 1);
            torch::nn::init::constant_(value_net_->bias, 0);
        }

        torch::Tensor PredictProbFromFeature(const torch::Tensor& feature) {
            torch::Tensor y = action_mlp_->forward(feature);
            y = torch::relu(y);
            y = action_net_->forward(y);
            y = torch::softmax(y, -1);
            return y;
        }

        torch::Tensor PredictValueFromFeature(const torch::Tensor& feature) {
            torch::Tensor y = value_mlp_->forward(feature);
            y = torch::relu(y);
            y = value_net_->forward(y).squeeze(-1);
            return y;
        }

        torch::Tensor PredictAction(const torch::Tensor& observation, bool deterministic = true) override {
            torch::Tensor feature = feature_extractor_->forward(observation);
            torch::Tensor prob = PredictProbFromFeature(feature);
            if (deterministic)
                return std::get<1>(torch::max(prob, -1));
            else
                return torch::multinomial(prob, 1).squeeze(-1);
        }

        torch::Tensor PredictValue(const torch::Tensor& observation) override {
            torch::Tensor feature = feature_extractor_->forward(observation);
            return PredictValueFromFeature(feature);
        }

        std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateAction(const torch::Tensor& observation, const torch::Tensor& action) override {
            torch::Tensor feature = feature_extractor_->forward(observation);
            torch::Tensor prob = PredictProbFromFeature(feature);
            torch::Tensor log_prob = torch::log(prob + 1e-8);
            torch::Tensor entropy = -torch::sum(prob * log_prob, -1);
            torch::Tensor value = PredictValueFromFeature(feature);
            torch::Tensor action_log_prob = torch::gather(log_prob, 1, action.reshape({-1, 1}));
            return { value, action_log_prob.squeeze(-1), { entropy } };
        }

        std::array<torch::Tensor, 3> Forward(const torch::Tensor& observation) override {
            torch::Tensor feature = feature_extractor_->forward(observation);
            torch::Tensor prob = PredictProbFromFeature(feature);
            torch::Tensor log_prob = torch::log(prob + 1e-8);
            torch::Tensor action = torch::multinomial(prob, 1);
            torch::Tensor action_log_prob = torch::gather(log_prob, 1, action);
            torch::Tensor value = PredictValueFromFeature(feature);
            return { action.squeeze(-1), value, action_log_prob.squeeze(-1) };
        }

    private:
        torch::nn::Sequential feature_extractor_;
        torch::nn::Linear action_mlp_, value_mlp_;
        torch::nn::Linear action_net_, value_net_;
    };
}