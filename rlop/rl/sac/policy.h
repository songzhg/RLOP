#pragma once
#include "rlop/rl/policy.h"

namespace rlop {
    // SACActor is an abstract class that defines the interface for the actor component of the Soft Actor-Critic algorithm.
    // The actor is responsible for selecting actions given observations from the environment, following a policy that seeks
    // to maximize the expected return while also maximizing entropy to encourage exploration.
    class SACActor : public RLPolicy {
    public:
        constexpr static double kLogStdMax = 2.0;
        constexpr static double kLogStdMin = -20.0;

        SACActor() = default;

        virtual ~SACActor() = default;

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
    };

    // SACCritic is an abstract class that defines the interface for the critic component of the Soft Actor-Critic algorithm.
    // The critic estimates the value of taking certain actions in certain states, helping the actor adjust its policy towards
    // actions that lead to higher returns. It typically consists of one or more neural networks that are trained to minimize
    // the temporal difference error.
    class SACCritic : public torch::nn::Module {
    public:
        virtual ~SACCritic() = default;

        virtual void Reset() = 0;

        // Performs a forward pass of the critic network(s) to estimate the value(s) of taking specific actions in specific states.
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //   action: A torch::Tensor representing the action taken by the actor in the given state.
        //
        // Returns:
        //   std::vector<torch::Tensor>: A vector of tensors representing the estimated values of the given actions in the given states.
        //   Depending on the SAC implementation, this might include Q-values from multiple critic networks.
        virtual std::vector<torch::Tensor> Forward(const torch::Tensor& observation, const torch::Tensor& action) = 0;
    };
}