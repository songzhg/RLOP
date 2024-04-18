#pragma once
#include "rlop/rl/ppo/policy.h"
#include "rlop/rl/distributions.h"

namespace continuous_lunar_lander {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class PPOPolicy : public rlop::PPOPolicy {
    public:
        PPOPolicy(Int observation_dim, Int action_dim) {
            action_mlp_->push_back(torch::nn::Linear(observation_dim, 64));
            action_mlp_->push_back(torch::nn::Tanh());
            action_mlp_->push_back(torch::nn::Linear(64, 64));
            action_mlp_->push_back(torch::nn::Tanh());
            value_mlp_->push_back(torch::nn::Linear(observation_dim, 64));
            value_mlp_->push_back(torch::nn::Tanh());
            value_mlp_->push_back(torch::nn::Linear(64, 64));
            value_mlp_->push_back(torch::nn::Tanh());
            register_module("action_mlp", action_mlp_);
            register_module("value_mlp", value_mlp_);
            action_net_ = register_module("action_net", torch::nn::Linear(64, action_dim));
            value_net_ = register_module("value_net", torch::nn::Linear(64, 1));
            log_std_ = register_parameter("log_std", torch::zeros(action_dim, torch::requires_grad(true)));
        }

        void Reset() override {
            rlop::PPOPolicy::Reset();
            action_mlp_->apply([](torch::nn::Module& module){ rlop::PPOPolicy::InitWeights(&module, std::sqrt(2.0)); });
            value_mlp_->apply([](torch::nn::Module& module){ rlop::PPOPolicy::InitWeights(&module, std::sqrt(2.0)); });
            action_net_->apply([](torch::nn::Module& module){ rlop::PPOPolicy::InitWeights(&module, 0.01); });
            value_net_->apply([](torch::nn::Module& module){ rlop::PPOPolicy::InitWeights(&module, 1.0); });
        }

        torch::Tensor PredictDist(const torch::Tensor& observations) {
            torch::Tensor latent_pi = action_mlp_->forward(observations);
            return action_net_->forward(latent_pi);
        }

        torch::Tensor PredictActions(const torch::Tensor& observations, bool deterministic = false) override {
            torch::Tensor mean = PredictDist(observations);
            rlop::DiagGaussian dist(mean, log_std_.exp());
            if (deterministic) 
                return dist.Mode();
            else {
                return dist.Sample();
            }
        }

        torch::Tensor PredictValues(const torch::Tensor& observations) override {
            torch::Tensor latent_pi = value_mlp_->forward(observations);
            return value_net_->forward(latent_pi).flatten();
        }

        std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions) override {
            torch::Tensor mean = PredictDist(observations);
            rlop::DiagGaussian dist(mean, log_std_.exp());
            torch::Tensor log_prob = dist.LogProb(actions);
            torch::Tensor entropy = dist.Entropy();
            torch::Tensor values = PredictValues(observations);
            return { values, log_prob, { entropy } };
        }

        std::array<torch::Tensor, 3> Forward(const torch::Tensor& observations) override {
            torch::Tensor mean = PredictDist(observations);
            rlop::DiagGaussian dist(mean, log_std_.exp());
            torch::Tensor actions = dist.Sample();
            torch::Tensor log_prob = dist.LogProb(actions);
            torch::Tensor values = PredictValues(observations);
            return { actions, values, log_prob };
        }

    private:
        torch::nn::Sequential action_mlp_;
        torch::nn::Sequential value_mlp_;
        torch::nn::Linear action_net_{nullptr};
        torch::nn::Linear value_net_{nullptr};
        torch::Tensor log_std_;
    };
}