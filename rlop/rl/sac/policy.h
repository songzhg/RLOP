#pragma once
#include "rlop/rl/policy.h"

namespace rlop {
    class ContinuousQNet : public torch::nn::Module {
    public:
        ContinuousQNet() = default;

        virtual ~ContinuousQNet() = default;

        // Performs a forward pass of the critic network(s) to estimate the value(s) of taking specific actions in specific states.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //   action: A torch::Tensor representing the action taken by the actor in the given state.
        //
        // Returns:
        //   std::vector<torch::Tensor>: A vector of tensors representing the estimated values of the given actions in the given states.
        //   Depending on the SAC implementation, this might include Q-values from multiple critic networks.
        virtual std::vector<torch::Tensor> PredictQValues(const torch::Tensor& observation, const torch::Tensor& action) = 0;
    };

    class SACPolicy : public RLPolicy {
    public:
        constexpr static double kLogStdMax = 2.0;
        constexpr static double kLogStdMin = -20.0;

        SACPolicy() = default;

        virtual ~SACPolicy() = default;

        // Predicts an action and its log probability for a given observation from the environment.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //
        // Returns:
        //   std::array<torch::Tensor, 2>: A tuple containing the predicted action and its log probability.
        //     - [0]: The predicted action as a torch::Tensor.
        //     - [1]: The log probability of the predicted action as a torch::Tensor.
        virtual std::array<torch::Tensor, 2> PredictLogProb(const torch::Tensor& observation) = 0;

        // Factory method to create and return a shared pointer to a critic object.
        virtual std::shared_ptr<ContinuousQNet> MakeCritic() const = 0;

        virtual void Reset() override {
            this->to(device_);
            critic_= MakeCritic();
            critic_->to(device_);
            critic_target_ = MakeCritic();
            critic_target_->to(device_);
            critic_target_->eval();
            torch_utils::CopyStateDict(*critic_, critic_target_.get());
        }

        virtual void SetTrainingMode(bool mode) override {
            if (mode) {
                this->train();
                critic_->train(); 
            }
            else {
                this->eval();
                critic_->eval(); 
            }
        }

        std::shared_ptr<ContinuousQNet> critic() const {
            return critic_;
        } 

        std::shared_ptr<ContinuousQNet> critic_target() const {
            return critic_target_;
        } 
        
    protected:
        std::shared_ptr<ContinuousQNet> critic_ = nullptr;
        std::shared_ptr<ContinuousQNet> critic_target_ = nullptr;
    };
}