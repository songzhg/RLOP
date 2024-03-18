#pragma once
#include "rlop/rl/policy.h"

namespace rlop {
    // QNet is an abstract base class for implementing Q-networks.
    // Q-networks estimate the value of taking actions in given states (or observations) by learning the Q-function.
    // This class extends RLPolicy, providing a common interface for predicting actions based on observed states.
    class QNet : public torch::nn::Module {
    public:
        QNet() = default;

        virtual ~QNet() = default;

        // Predict the Q-value of each action given an observation.
        // This method must be implemented by derived classes to specify the network's architecture and behavior.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //
        // Returns:
        //   torch::Tensor: The estimated Q-values for all possible actions given the input observation.
        virtual torch::Tensor PredictQValues(const torch::Tensor& observations) = 0;
    };

    class DQNPolicy : public RLPolicy {
    public:
        DQNPolicy() = default;

        virtual ~DQNPolicy() = default;

        // Factory method to create and return a shared pointer to a Q-networks.
        virtual std::shared_ptr<QNet> MakeQNet() const = 0;

        // Predicts the action for a given state (or observation) using the Q-network.
        // By default, it selects the action with the highest estimated Q-value (greedy action).
        // This behavior can be overridden in derived classes if needed.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //   deterministic: A boolean flag indicating whether the action selection should be deterministic.
        //                  If true, the action with the highest Q-value is selected. This flag is included
        //                  for compatibility with stochastic policies but is not used in the base implementation.
        //
        // Returns:
        //   torch::Tensor: The selected action as a tensor. The action corresponds to the index of the
        //                  highest Q-value predicted by the network for the given observation.
        virtual torch::Tensor PredictActions(const torch::Tensor& observation, bool deterministic = false) override {
            return std::get<1>(torch::max(q_net_->PredictQValues(observation), -1));
        }

        virtual void Reset() override {
            q_net_= MakeQNet();
            q_net_->to(device_);
            q_net_target_ = MakeQNet();
            q_net_target_->to(device_);
            q_net_target_->eval();
            torch_utils::CopyStateDict(*q_net_, q_net_target_.get());
        }

        virtual void SetTrainingMode(bool mode) override {
            if (mode)
                q_net_->train();
            else 
                q_net_->eval();
        }

        std::shared_ptr<QNet> q_net() const {
            return q_net_;
        } 

        std::shared_ptr<QNet> q_net_target() const {
            return q_net_target_;
        } 

    protected:
        std::shared_ptr<QNet> q_net_ = nullptr;
        std::shared_ptr<QNet> q_net_target_ = nullptr;
    };
}