#pragma once
#include "rlop/rl/policy.h"

namespace rlop {
    // PPOPolicy is an abstract class that defines the interface for policy networks used in the Proximal Policy Optimization (PPO) algorithm.
    // It extends the RLPolicy class, providing specialized methods required for the PPO algorithm, such as value prediction and action evaluation.
    class PPOPolicy : public RLPolicy {
    public:
        PPOPolicy() = default;

        virtual ~PPOPolicy() = default;

        // Predicts the value of a given state or observation. This method is typically used to estimate the expected return from a given state,
        // which is crucial for computing advantages during PPO updates.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //
        // Returns:
        //   torch::Tensor: The predicted value of the state as a tensor.
        virtual torch::Tensor PredictValues(const torch::Tensor& observations) = 0;

        // Evaluates a given action taken in a given state or observation. This method is used to compute log probabilities, values, and optionally,
        // entropy or action probabilities, which are essential for the PPO update step.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //   action: A torch::Tensor representing the action taken in the given state.
        //
        // Returns:
        //   std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>>: A tuple containing:
        //     - [0]: The value of the state as predicted by the policy's value function.
        //     - [1]: The log probability of taking the given action in the given state.
        //     - [2]: An optional tensor representing additional information such as entropy of the action distribution (if applicable).
        virtual std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> EvaluateActions(const torch::Tensor& observations, const torch::Tensor& actions) = 0;

        // Performs a forward pass of the policy network given an observation. This method is typically used to generate actions
        // from states during interaction with the environment.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //
        // Returns:
        //   std::array<torch::Tensor, 3>: An array containing:
        //     - [0]: The action recommended by the policy for the given observation.
        //     - [1]: The predicted value of the current state.
        //     - [2]: The log probability of the recommended action.
        virtual std::array<torch::Tensor, 3> Forward(const torch::Tensor& observations) = 0;
    };
}