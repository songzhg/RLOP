#pragma once
#include "rlop/rl/sac/policy.h"
#include "rlop/rl/distributions.h"

namespace continuous_lunar_lander {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    Int kWidth = 256;

    class SACCritic : public rlop::ContinuousQNet {
    public:
        SACCritic(Int num_critics, Int observation_dim, Int action_dim) : q_nets_(num_critics) {
            for (Int i=0; i<num_critics; ++i) {
                q_nets_[i]->push_back(torch::nn::Linear(observation_dim + action_dim, kWidth));
                q_nets_[i]->push_back(torch::nn::ReLU());
                q_nets_[i]->push_back(torch::nn::Linear(kWidth, kWidth));
                q_nets_[i]->push_back(torch::nn::ReLU());
                q_nets_[i]->push_back(torch::nn::Linear(kWidth, 1));
                register_module("q_net_" + std::to_string(i), q_nets_[i]);
            }
        }

        std::vector<torch::Tensor> PredictQValues(const torch::Tensor& observations, const torch::Tensor& actions) override {
            std::vector<torch::Tensor> q_values;
            q_values.reserve(q_nets_.size()); 
            torch::Tensor input = torch::cat({observations, actions}, 1);
            for (Int i=0; i<q_nets_.size(); ++i) {
                q_values.push_back(q_nets_[i]->forward(input));
            }
            return q_values;
        }

    private:
        std::vector<torch::nn::Sequential> q_nets_;
    };

    class SACPolicy : public rlop::SACPolicy {
    public:
        SACPolicy(Int observation_dim, Int action_dim, Int num_critics) :
            observation_dim_(observation_dim),
            action_dim_(action_dim),
            num_critics_(num_critics)
        {
            latent_pi_->push_back(torch::nn::Linear(observation_dim, 256));
            latent_pi_->push_back(torch::nn::ReLU());
            latent_pi_->push_back(torch::nn::Linear(256, 256));
            latent_pi_->push_back(torch::nn::ReLU());
            register_module("latent_pi", latent_pi_);
            mu_ = register_module("mu", torch::nn::Linear(256, action_dim));
            log_std_ = register_module("log_std", torch::nn::Linear(256, action_dim));
        }

        std::shared_ptr<rlop::ContinuousQNet> MakeCritic() const override {
            return std::make_shared<SACCritic>(num_critics_, observation_dim_, action_dim_);
        }

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
            rlop::SquashedDiagGaussian dist(mean, log_std.exp());
            if (deterministic)
                return dist.Mode();
            return dist.Sample();
        }
        
    private:
        torch::nn::Sequential latent_pi_;
        torch::nn::Linear mu_{nullptr};
        torch::nn::Linear log_std_{nullptr};
        Int observation_dim_;
        Int action_dim_;
        Int num_critics_;
    };
}