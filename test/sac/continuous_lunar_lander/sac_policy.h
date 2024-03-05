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
            latent_pi_(
                torch::nn::Linear(observation_dim, 256),
                torch::nn::ReLU(),
                torch::nn::Linear(256, 256),
                torch::nn::ReLU()
            ),
            mu_(torch::nn::Linear(256, action_dim)),
            log_std_(torch::nn::Linear(256, action_dim))
        {
            register_module("latent_pi", latent_pi_);
            register_module("mu", mu_);
            register_module("log_std", log_std_);
        }

        void Reset() override {}

        std::array<torch::Tensor, 2> PredictDist(const torch::Tensor& observations) {
            torch::Tensor y = latent_pi_->forward(observations);
            torch::Tensor mean = mu_->forward(y);
            torch::Tensor log_std = log_std_->forward(y);
            log_std = torch::clamp(log_std, kLogStdMin, kLogStdMax);
            return { mean, log_std };
        }

        std::array<torch::Tensor, 2> PredictLogProb(const torch::Tensor& observations) override {
            auto [ mean, log_std ] = PredictDist(observations);
            rlop::SquashedDiagGaussian dist(mean, log_std.exp());
            torch::Tensor gaussian_actions = dist.rlop::DiagGaussian::Sample();
            torch::Tensor actions = torch::tanh(gaussian_actions);
            torch::Tensor log_prob = dist.LogProb(actions, gaussian_actions);
            return { actions, log_prob };
        }

        torch::Tensor PredictActions(const torch::Tensor& observations, bool deterministic = false) override {
            auto [ mean, log_std ] = PredictDist(observations);
            if (deterministic)
                return mean;
            rlop::SquashedDiagGaussian dist(mean, log_std.exp());
            return dist.Sample();
        }
        
    private:
        torch::nn::Sequential latent_pi_;
        torch::nn::Linear mu_;
        torch::nn::Linear log_std_;
    };

    class SACCritic : public rlop::SACCritic {
    public:
        SACCritic(Int num_critics, Int observation_dim, Int action_dim) {
            q_nets_.reserve(num_critics);
            for (Int i=0; i<num_critics; ++i) {
                q_nets_.emplace_back(
                    torch::nn::Linear(observation_dim + action_dim, 256),
                    torch::nn::ReLU(),
                    torch::nn::Linear(256, 256),
                    torch::nn::ReLU(),
                    torch::nn::Linear(256, 1)
                );
                register_module("q_net_" + std::to_string(i), q_nets_[i]);
            }
        }

        void Reset() override {}

        std::vector<torch::Tensor> Forward(const torch::Tensor& observations, const torch::Tensor& actions) override {
            std::vector<torch::Tensor> q_values;
            q_values.reserve(q_nets_.size()); 
            torch::Tensor input = torch::cat({observations, actions}, 1);
            for (Int i=0; i<q_nets_.size(); ++i) {
                q_values.push_back(q_nets_[i]->forward(input).flatten());
            }
            return q_values;
        }

    private:
        std::vector<torch::nn::Sequential> q_nets_;
    };
}