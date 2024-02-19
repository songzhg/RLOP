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
            torch::nn::init::orthogonal_(value_net_->weight, 1);
            torch::nn::init::constant_(action_net_->bias, 0);
            torch::nn::init::constant_(value_net_->bias, 0);
        }

        torch::Tensor PredictDist(const torch::Tensor& observation) {
            torch::Tensor y = action_mlp_->forward(observation);
            y = action_net_->forward(y);
            return torch::softmax(y, -1);
        }

        torch::Tensor PredictAction(const torch::Tensor& observation, bool deterministic = true) override {
            torch::Tensor dist = PredictDist(observation);
            if (deterministic)
                return std::get<1>(torch::max(dist, -1));
            else
                return torch::multinomial(dist, 1).squeeze(-1);
        }

        torch::Tensor PredictValue(const torch::Tensor& observation) override {
            torch::Tensor y = value_mlp_->forward(observation);
            return value_net_->forward(y).squeeze(-1);
        }

        std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateAction(const torch::Tensor& observation, const torch::Tensor& action) override {
            torch::Tensor dist = PredictDist(observation);
            torch::Tensor log_prob = torch::log(dist + 1e-8);
            torch::Tensor entropy = -torch::sum(dist * log_prob, -1);
            torch::Tensor value = PredictValue(observation);
            torch::Tensor action_log_prob = torch::gather(log_prob, 1, action.reshape({-1, 1}));
            return { value, action_log_prob.squeeze(-1), { entropy } };
        }

        std::array<torch::Tensor, 3> Forward(const torch::Tensor& observation) override {
            torch::Tensor prob = PredictDist(observation);
            torch::Tensor log_prob = torch::log(prob + 1e-8);
            torch::Tensor action = torch::multinomial(prob, 1);
            torch::Tensor action_log_prob = torch::gather(log_prob, 1, action);
            torch::Tensor value = PredictValue(observation);
            return { action.squeeze(-1), value, action_log_prob.squeeze(-1) };
        }

    private:
        torch::nn::Sequential action_mlp_;
        torch::nn::Sequential value_mlp_;
        torch::nn::Linear action_net_, value_net_;
    };
}