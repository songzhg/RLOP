#pragma once
#include "rlop/rl/ppo/policy.h"

namespace lunar_lander {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class PPOPolicy : public rlop::PPOPolicy {
    public:
        PPOPolicy(Int observation_dim, Int num_actions) :
            action_mlp_(
                torch::nn::Linear(observation_dim, 64),
                torch::nn::Tanh(),
                torch::nn::Linear(64, 64),
                torch::nn::Tanh()
            ),
            value_mlp_(
                torch::nn::Linear(observation_dim, 64),
                torch::nn::Tanh(),
                torch::nn::Linear(64, 64),
                torch::nn::Tanh()
            ),
            action_net_(torch::nn::Linear(64, num_actions)),
            value_net_(torch::nn::Linear(64, 1))
        {
            register_module("action_mlp", action_mlp_);
            register_module("value_mlp", value_mlp_);
            register_module("action_net", action_net_);
            register_module("value_net", value_net_);
        }

        void Reset() override {
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

        torch::Tensor PredictDist(const torch::Tensor& observations) {
            torch::Tensor latent_pi = action_mlp_->forward(observations);
            latent_pi = action_net_->forward(latent_pi);
            return torch::softmax(latent_pi, -1);
        }

        torch::Tensor PredictActions(const torch::Tensor& observations, bool deterministic = true) override {
            torch::Tensor dist = PredictDist(observations);
            if (deterministic)
                return std::get<1>(torch::max(dist, -1));
            else
                return torch::multinomial(dist, 1).flatten();
        }

        torch::Tensor PredictValues(const torch::Tensor& observations) override {
            torch::Tensor latent_pi = value_mlp_->forward(observations);
            return value_net_->forward(latent_pi).flatten();
        }

        std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions) override {
            torch::Tensor dist = PredictDist(observations);
            torch::Tensor log_prob = torch::log(dist + 1e-8);
            torch::Tensor entropy = -torch::sum(dist * log_prob, -1);
            torch::Tensor values = PredictValues(observations);
            torch::Tensor action_log_prob = torch::gather(log_prob, 1, actions.reshape({-1, 1}));
            return { values, action_log_prob.flatten(), { entropy } };
        }

        std::array<torch::Tensor, 3> Forward(const torch::Tensor& observations) override {
            torch::Tensor prob = PredictDist(observations);
            torch::Tensor log_prob = torch::log(prob + 1e-8);
            torch::Tensor actions = torch::multinomial(prob, 1);
            torch::Tensor action_log_prob = torch::gather(log_prob, 1, actions);
            torch::Tensor values = PredictValues(observations);
            return { actions.flatten(), values, action_log_prob.flatten() };
        }

    private:
        torch::nn::Sequential action_mlp_;
        torch::nn::Sequential value_mlp_;
        torch::nn::Linear action_net_, value_net_;
    };
}