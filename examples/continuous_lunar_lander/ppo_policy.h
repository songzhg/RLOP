#pragma once
#include "rlop/rl/ppo/policy.h"
#include "rlop/rl/distributions.h"

namespace continuous_lunar_lander {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class PPOPolicy : public rlop::PPOPolicy {
    public:
        PPOPolicy(Int observation_size, Int action_dim) :
            action_mlp_(
                torch::nn::Linear(observation_size, 64),
                torch::nn::Tanh(),
                torch::nn::Linear(64, 64),
                torch::nn::Tanh()
            ),
            value_mlp_(
                torch::nn::Linear(observation_size, 64),
                torch::nn::Tanh(),
                torch::nn::Linear(64, 64),
                torch::nn::Tanh()
            ),
            action_net_(torch::nn::Linear(64, action_dim)),
            value_net_(torch::nn::Linear(64, 1)),
            log_std_(torch::zeros(action_dim, torch::requires_grad(true)))
        {
            register_module("action_mlp", action_mlp_);
            register_module("value_mlp", value_mlp_);
            register_module("action_net", action_net_);
            register_module("value_net", value_net_);
            register_parameter("log_std", log_std_);
        }

        void Reset() override {
            for (auto& module : action_mlp_->modules()) {
                auto* linear = dynamic_cast<torch::nn::LinearImpl*>(module.get());
                if (linear) {
                    torch::nn::init::orthogonal_(linear->weight, std::sqrt(2));
                    torch::nn::init::constant_(linear->bias, 0);
                }
            }
            for (auto& module : value_mlp_->modules()) {
                auto* linear = dynamic_cast<torch::nn::LinearImpl*>(module.get());
                if (linear) {
                    torch::nn::init::orthogonal_(linear->weight, std::sqrt(2));
                    torch::nn::init::constant_(linear->bias, 0);
                }
            }
            torch::nn::init::orthogonal_(action_net_->weight, 0.01);
            torch::nn::init::orthogonal_(value_net_->weight, 1.0);
            torch::nn::init::constant_(action_net_->bias, 0);
            torch::nn::init::constant_(value_net_->bias, 0);
            torch::nn::init::constant_(log_std_, 0);
        }

        torch::Tensor PredictDist(const torch::Tensor& observation) {
            torch::Tensor y = action_mlp_->forward(observation);
            return action_net_->forward(y);
        }

        torch::Tensor PredictAction(const torch::Tensor& observation, bool deterministic = true) override {
            torch::Tensor mean = PredictDist(observation);
            if (deterministic) 
                return mean;
            else {
                torch::Tensor log_std = torch::ones_like(mean) * log_std_;
                rlop::DiagGaussian dist(mean, log_std);
                return dist.Sample(mean.sizes());
            }
        }

        torch::Tensor PredictValue(const torch::Tensor& observation) override {
            torch::Tensor y = value_mlp_->forward(observation);
            return value_net_->forward(y).squeeze(-1);
        }

        std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateAction(const torch::Tensor& observation, const torch::Tensor& action) override {
            torch::Tensor mean = PredictDist(observation);
            torch::Tensor log_std = torch::ones_like(mean) * log_std_;
            rlop::DiagGaussian dist(mean, log_std);
            torch::Tensor action_log_prob = dist.LogProb(action).sum(1);
            torch::Tensor entropy = dist.Entropy().sum(1);
            torch::Tensor value = PredictValue(observation);
            return { value, action_log_prob, { entropy } };
        }

        std::array<torch::Tensor, 3> Forward(const torch::Tensor& observation) override {
            torch::Tensor mean = PredictDist(observation);
            torch::Tensor log_std = torch::ones_like(mean) * log_std_;
            rlop::DiagGaussian dist(mean, log_std);
            torch::Tensor action = dist.Sample(mean.sizes());
            torch::Tensor action_log_prob = dist.LogProb(action).sum(1);
            torch::Tensor value = PredictValue(observation);
            return { action, value, action_log_prob };
        }

    private:
        torch::nn::Sequential action_mlp_;
        torch::nn::Sequential value_mlp_;
        torch::nn::Linear action_net_, value_net_;
        torch::Tensor log_std_;
    };
}