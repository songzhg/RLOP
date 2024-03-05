#pragma once
#include "rlop/rl/policy.h"

namespace rlop {
    // QNet is an abstract base class for implementing Q-networks in reinforcement learning algorithms.
    // Q-networks estimate the value of taking actions in given states (or observations) by learning the Q-function.
    // This class extends RLPolicy, providing a common interface for predicting actions based on observed states.
    class QNet : public RLPolicy {
    public:
        QNet() = default;

        virtual ~QNet() = default;

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
            return std::get<1>(torch::max(Forward(observation), -1));
        }

        // Performs a forward pass through the Q-network given an observation.
        // This method must be implemented by derived classes to specify the network's architecture and behavior.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //
        // Returns:
        //   torch::Tensor: The estimated Q-values for all possible actions given the input observation.
        virtual torch::Tensor Forward(const torch::Tensor& observations) = 0;
    };
}