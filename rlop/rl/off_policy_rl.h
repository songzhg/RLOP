#pragma once
#include "rl.h"
#include "buffers.h"
#include "policy.h"

namespace rlop {
    class OffPolicyRL : public RL {
    public:
        OffPolicyRL(
            Int learning_starts,
            Int train_freq,
            const std::string& output_path, 
            const torch::Device& device
        ) :
            learning_starts_(learning_starts),
            train_freq_(train_freq),
            RL(output_path, device) 
        {}

        virtual ~OffPolicyRL() = default;

        // Factory method to create and return a shared pointer to a replay buffer.
        virtual std::shared_ptr<ReplayBuffer> MakeReplayBuffer() const = 0; 

        // Factory method to create and return a shared pointer to a policy object.
        virtual std::shared_ptr<RLPolicy> MakePolicy() const = 0; 

        // Pure virtual function to sample an action from the action space.
        //
        // Returns:
        //   torch::Tensor: A tensor representing the selected actions.
        virtual torch::Tensor SampleActions() = 0;

        virtual void Reset() override {
            RL::Reset();
            replay_buffer_ = MakeReplayBuffer();
            replay_buffer_->Reset();
            policy_ = MakePolicy();
            policy_->To(device_);
            policy_->Reset();
            last_observations_ = ResetEnv();
        }

        virtual void StoreTransition(
            const torch::Tensor& actions, 
            const torch::Tensor& new_observations, 
            const torch::Tensor& rewards, 
            const torch::Tensor& terminations,
            const torch::Tensor& truncations,
            const torch::Tensor& final_observations
        ) {
            torch::Tensor next_observations = new_observations.clone();
            if (final_observations.defined()) {
                torch::Tensor dones = torch::logical_or(terminations, truncations);
                for (Int i=0; i<replay_buffer_->num_envs(); ++i) {
                    if (dones[i].item<bool>()) 
                        next_observations[i].copy_(final_observations[i]);
                }
            }
            replay_buffer_->Add(last_observations_, actions, next_observations, rewards, terminations); 
            last_observations_ = new_observations;
        }

        virtual void OnCollectRolloutStep() {}

        virtual void CollectRollouts() override {
            policy_->SetTrainingMode(false);
            torch::NoGradGuard no_grad;
            for (Int step = 0; step < train_freq_; ++step) {
                torch::Tensor actions;
                if (time_steps_ < learning_starts_)
                    actions = SampleActions();
                else 
                    actions = Predict(last_observations_, false)[0];
                auto [new_observations, rewards, terminations, truncations, final_observations] = Step(actions);
                time_steps_ += replay_buffer_->num_envs();
                StoreTransition(actions, new_observations, rewards, terminations, truncations, final_observations);
                OnCollectRolloutStep();
            }
        }

        virtual std::array<torch::Tensor, 2> Predict(const torch::Tensor& observation, bool deterministic = false, const torch::Tensor& state = torch::Tensor(), const torch::Tensor& episode_start = torch::Tensor()) override {
            return policy_->Predict(observation, deterministic, state, episode_start);
        }

        virtual void Monitor() override {
            if (time_steps_ <= learning_starts_)
                return;
            RL::Monitor();
        }

        std::shared_ptr<ReplayBuffer> replay_buffer() const {
            return replay_buffer_;
        }

        std::shared_ptr<RLPolicy> policy() const {
            return policy_;
        }

    protected:
        Int learning_starts_;
        Int train_freq_;
        torch::Tensor last_observations_;
        std::shared_ptr<ReplayBuffer> replay_buffer_ = nullptr;
        std::shared_ptr<RLPolicy> policy_ = nullptr;
    };
}