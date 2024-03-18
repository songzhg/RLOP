#pragma once
#include "rlop/rl/ppo/policy.h"
#include "rlop/rl/distributions.h"

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
            rlop::PPOPolicy::Reset();
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

        torch::Tensor PredictActionLogits(const torch::Tensor& observations) {
            torch::Tensor latent_pi = action_mlp_->forward(observations);
            return action_net_->forward(latent_pi);
        }

        torch::Tensor PredictActions(const torch::Tensor& observations, bool deterministic = false) override {
            torch::Tensor logits = PredictActionLogits(observations);
            rlop::Categorical dist(logits);
            if (deterministic)
                return dist.Mode();
            else
                return dist.Sample();
        }

        torch::Tensor PredictValues(const torch::Tensor& observations) override {
            torch::Tensor latent_pi = value_mlp_->forward(observations);
            return value_net_->forward(latent_pi).flatten();
        }

        std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions) override {
            torch::Tensor logits = PredictActionLogits(observations);
            torch::Tensor values = PredictValues(observations);
            rlop::Categorical dist(logits);
            torch::Tensor log_prob = dist.LogProb(actions);
            torch::Tensor entropy = dist.Entropy();
            return { values, log_prob, { entropy } };
        }

        std::array<torch::Tensor, 3> Forward(const torch::Tensor& observations) override {
            torch::Tensor logits = PredictActionLogits(observations);
            rlop::Categorical dist(logits);
            torch::Tensor actions = dist.Sample();
            torch::Tensor values = PredictValues(observations);
            torch::Tensor log_prob = dist.LogProb(actions);
            return { actions, values, log_prob };
        }

    private:
        torch::nn::Sequential action_mlp_;
        torch::nn::Sequential value_mlp_;
        torch::nn::Linear action_net_, value_net_;
    };
}