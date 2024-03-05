#pragma once
#include "rlop/rl/ppo/policy.h"

namespace snake {
    class PPOPolicy : public rlop::PPOPolicy {
    public:
        PPOPolicy(const std::vector<Int>& observation_sizes, Int num_actions) :
            observation_sizes_(observation_sizes),
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
                rlop::RLPolicy::InitWeights(module.get(), std::sqrt(2.0));
            }
            for (auto& module : action_mlp_->modules()) {
                rlop::RLPolicy::InitWeights(module.get(), std::sqrt(2.0));
            }
            for (auto& module : value_mlp_->modules()) {
                rlop::RLPolicy::InitWeights(module.get(), std::sqrt(2.0));
            }
            for (auto& module : action_net_->modules()) {
                rlop::RLPolicy::InitWeights(module.get(), 0.01);
            }
            for (auto& module : value_net_->modules()) {
                rlop::RLPolicy::InitWeights(module.get(), std::sqrt(1.0));
            }
        }

        torch::Tensor ExtractFeatures(const torch::Tensor& observations) {
            if (observations.sizes().size() == observation_sizes_.size())
                return feature_extractor_->forward(observations.unsqueeze(0));
            else
                return feature_extractor_->forward(observations);
        }

        torch::Tensor PredictProbFromFeature(const torch::Tensor& features) {
            torch::Tensor latent_pi = torch::relu(action_mlp_->forward(features));
            return torch::softmax(action_net_->forward(latent_pi), -1);
        }

        torch::Tensor PredictValueFromFeature(const torch::Tensor& features) {
            torch::Tensor latent_vf = torch::relu(value_mlp_->forward(features));
            return value_net_->forward(latent_vf).squeeze(-1);
        }

        torch::Tensor PredictActions(const torch::Tensor& observations, bool deterministic = true) override {
            torch::Tensor features = ExtractFeatures(observations);
            torch::Tensor prob = PredictProbFromFeature(features);
            if (deterministic)
                return std::get<1>(torch::max(prob, -1));
            else
                return torch::multinomial(prob, 1).squeeze(-1);
        }

        torch::Tensor PredictValues(const torch::Tensor& observations) override {
            torch::Tensor features = ExtractFeatures(observations);
            return PredictValueFromFeature(features);
        }

        std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions) override {
            torch::Tensor features = ExtractFeatures(observations);
            torch::Tensor prob = PredictProbFromFeature(features);
            torch::Tensor log_prob = torch::log(prob + 1e-8);
            torch::Tensor entropy = -torch::sum(prob * log_prob, -1);
            torch::Tensor values = PredictValueFromFeature(features);
            torch::Tensor action_log_prob = torch::gather(log_prob, 1, actions.reshape({-1, 1}));
            return { values, action_log_prob.flatten(), { entropy } };
        }

        std::array<torch::Tensor, 3> Forward(const torch::Tensor& observations) override {
            torch::Tensor features = ExtractFeatures(observations);
            torch::Tensor prob = PredictProbFromFeature(features);
            torch::Tensor log_prob = torch::log(prob + 1e-8);
            torch::Tensor actions = torch::multinomial(prob, 1);
            torch::Tensor action_log_prob = torch::gather(log_prob, 1, actions);
            torch::Tensor values = PredictValueFromFeature(features);
            return { actions.flatten(), values, action_log_prob.flatten() };
        }

    private:
        torch::nn::Sequential feature_extractor_;
        torch::nn::Linear action_mlp_, value_mlp_;
        torch::nn::Linear action_net_, value_net_;
        std::vector<Int> observation_sizes_;
    };
}