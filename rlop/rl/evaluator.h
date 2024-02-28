#pragma once
#include "rlop/common/torch_utils.h"
#include "rlop/common/utils.h"

namespace rlop {
    // RLEvaluator is a classs to assess the performance of a RL model by executing it in
    // the environment for a given number of episodes and collecting statistics about rewards 
    // and episode lengths.
    class RLEvaluator {
    public:
        RLEvaluator() = default;

        virtual ~RLEvaluator() = default;

        virtual void Reset() {
            episode_rewards_.clear();
            episode_lengths_.clear();
        }

        virtual std::array<double, 2> Evaluate(RL* rl, Int num_eval_episodes, bool deterministic = true) {
            Int num_envs = rl->NumEnvs();
            episode_rewards_.reserve(episode_rewards_.size() + num_eval_episodes);
            episode_lengths_.reserve(episode_lengths_.size() + num_eval_episodes);
            std::vector<Int> episode_counts(num_envs, 0);
            std::vector<Int> episode_count_targets(num_envs);
            for (Int i = 0; i < num_envs; ++i) {
                episode_count_targets[i] = (num_eval_episodes + i) / num_envs;
            }
            torch::Tensor current_rewards = torch::zeros(num_envs);
            std::vector<Int> current_lengths(num_envs, 0);
            torch::Tensor state = torch::Tensor();
            torch::Tensor episode_start = torch::ones(num_envs, torch::kBool);
            torch::Tensor observation = rl->ResetEnv();
            while (true) {
                bool targets_met = true;
                for (Int i=0; i<num_envs; ++i) {
                    if (episode_counts[i] < episode_count_targets[i]) {
                        targets_met = false;
                        break;
                    }
                }
                if (targets_met)
                    break;
                auto [action, next_state] = rl->Predict(observation, deterministic, state, episode_start);
                auto [next_observation, reward, terminated, truncated, terminal_observation] = rl->Step(action);
                torch::Tensor done = torch::logical_or(terminated, truncated).to(torch::kFloat32);
                current_rewards += reward;
                std::for_each(current_lengths.begin(), current_lengths.end(), [](Int& item) { item += 1; });
                for (Int i = 0; i < num_envs; ++i) {
                    if (episode_counts[i] < episode_count_targets[i] && done[i].item<double>()) {
                        episode_rewards_.push_back(current_rewards[i].item<double>());
                        episode_lengths_.push_back(current_lengths[i]);
                        current_rewards[i] = 0;
                        current_lengths[i] = 0;
                        episode_counts[i] += 1;
                    }
                }
                observation = next_observation;
                state = next_state;
            }
            torch::Tensor reward = torch::tensor(episode_rewards_);
            return { reward.mean().item<double>(), reward.std().item<double>() };
        }

        const std::vector<float>& episode_rewards() const {
            return episode_rewards_;
        }

        const std::vector<Int>& episode_lengths() const {
            return episode_lengths_;
        }

    protected:
        std::vector<float> episode_rewards_;
        std::vector<Int> episode_lengths_;
    };
}