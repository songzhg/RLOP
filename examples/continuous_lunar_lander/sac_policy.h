#pragma once
#include "rlop/rl/sac/policy.h"
#include "rlop/rl/distributions.h"

namespace continuous_lunar_lander {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class SACActor : public rlop::SACActor {
    public:
        SACActor(Int observation_dim, Int action_dim) :
            mlp_(
                torch::nn::Linear(observation_dim, 256),
                torch::nn::ReLU(),
                torch::nn::Linear(256, 256),
                torch::nn::ReLU()
            ),
            mu_(torch::nn::Linear(256, action_dim)),
            log_std_(torch::nn::Linear(256, action_dim))
        {
            register_module("mlp", mlp_);
            register_module("mu", mu_);
            register_module("log_std", log_std_);
        }

        void Reset() override {
            for (auto& module : children()) {
                if (auto linear = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
                    torch::nn::init::xavier_uniform_(linear->weight);
                    torch::nn::init::constant_(linear->bias, 0);
                }
            }
        }

        std::array<torch::Tensor, 2> PredictDist(const torch::Tensor& observation) {
            torch::Tensor y = mlp_->forward(observation);
            torch::Tensor mean = mu_->forward(y);
            torch::Tensor log_std = log_std_->forward(y);
            log_std = torch::clamp(log_std, log_std_min_, log_std_max_);
            return { mean, log_std };
        }

        std::array<torch::Tensor, 2> PredictActionLogProb(const torch::Tensor& observation) override {
            auto [ mean, log_std ] = PredictDist(observation);
            rlop::SquashedDiagGaussian dist(mean, log_std);
            torch::Tensor action = dist.Sample(mean.sizes());
            torch::Tensor action_log_prob = dist.LogProb(action).sum(1);
            return { action, action_log_prob };
        }

        torch::Tensor PredictAction(const torch::Tensor& observation, bool deterministic = false) override {
            auto [ mean, log_std ] = PredictDist(observation);
            if (deterministic)
                return mean;
            rlop::SquashedDiagGaussian dist(mean, log_std);
            return dist.Sample(mean.sizes());
        }
        
    private:
        torch::nn::Sequential mlp_;
        torch::nn::Linear mu_, log_std_;
        double log_std_max_ = 2;
        double log_std_min_ = -20;
    };

    class SACCritic : public rlop::SACCritic {
    public:
        SACCritic(Int num_critics, Int observation_dim, Int action_dim) {
            mlps_.reserve(num_critics);
            for (Int i=0; i<num_critics; ++i) {
                mlps_.emplace_back(
                    torch::nn::Linear(observation_dim + action_dim, 256),
                    torch::nn::ReLU(),
                    torch::nn::Linear(256, 256),
                    torch::nn::ReLU(),
                    torch::nn::Linear(256, 1)
                );
                register_module("mlp" + std::to_string(i), mlps_[i]);
            }
        }

        void Reset() override {
            for (auto& module : children()) {
                if (auto linear = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
                    torch::nn::init::xavier_uniform_(linear->weight);
                    torch::nn::init::constant_(linear->bias, 0);
                }
            }
        }

        std::vector<torch::Tensor> Forward(const torch::Tensor& observation, const torch::Tensor& action) override {
            std::vector<torch::Tensor> q_values;
            q_values.reserve(mlps_.size()); 
            torch::Tensor input = torch::cat({observation, action}, 1);
            for (Int i=0; i<mlps_.size(); ++i) {
                q_values.push_back(mlps_[i]->forward(input).squeeze(-1));
            }
            return q_values;
        }

    private:
        std::vector<torch::nn::Sequential> mlps_;
    };
}