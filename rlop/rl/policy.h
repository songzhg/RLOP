#pragma once
#include "rlop/common/torch_utils.h"

namespace rlop {
    class RLPolicy : public torch::nn::Module {
    public:
        RLPolicy() = default;

        virtual ~RLPolicy() = default;

        virtual void Reset() = 0;

        // Pure virtual function to predict the action for a given state (or observation).
        //
        // Parameters:
        //   observation: A torch::Tensor representing the current state or observation from the environment.
        //   deterministic: A boolean flag indicating whether the action selection should be deterministic.
        //
        // Returns:
        //   torch::Tensor: The selected action as a tensor.
        virtual torch::Tensor PredictActions(const torch::Tensor& observations, bool deterministic = false) = 0;

        // Initializes the weights of linear and convolutional layers in a model using orthogonal initialization.
        //
        // Parameters:
        //   model: Pointer to the model.
        //   gain:  A scaling factor of weights .
        inline static void InitWeights(torch::nn::Module* module, double gain = 1.0) {
            module->apply([gain](torch::nn::Module& m) {
                if (auto* linear = m.as<torch::nn::Linear>()) {
                    torch::nn::init::orthogonal_(linear->weight, gain);
                    if (linear->bias.defined())
                        torch::nn::init::constant_(linear->bias, 0.0);
                }
                else if (auto* conv2d = m.as<torch::nn::Conv2d>()) {
                    torch::nn::init::orthogonal_(conv2d->weight, gain);
                    if (conv2d->bias.defined())
                        torch::nn::init::constant_(conv2d->bias, 0.0);
                }
            });
        }

        // Get the policy action from an observation (and optional hidden state). Includes sugar-coating to handle different observations
        // (e.g. normalizing images).
        //
        // Parameters:
        // observation: the input observation
        //   param state: The last hidden states (can be None, used in recurrent policies)
        //   episode_start: The last masks (can be None, used in recurrent policies) this correspond to beginning of episodes, where the 
        //                  hidden states of the RNN must be reset.
        //     
        //   param deterministic: Whether or not to return deterministic actions.
        //
        // Returns: 
        //   std::array<torch::Tensor, 3>: An array containing:
        //     - [0]: The model's action recommended by the policy for the given observation.
        //     - [1]: The next hidden state (used in recurrent policies)
        virtual std::array<torch::Tensor, 2> Predict(const torch::Tensor& observation, bool deterministic = false, const torch::Tensor& state = torch::Tensor(), const torch::Tensor& episode_start = torch::Tensor()) {
            this->eval();
            torch::NoGradGuard no_grad;
            return { PredictActions(observation, deterministic), torch::Tensor() };
        }
    };
}