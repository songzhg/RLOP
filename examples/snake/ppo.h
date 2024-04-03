#pragma once
#include "problems/snake/problem.h"
#include "ppo_policy.h"
#include "rlop/rl/ppo/ppo.h"
#include "rlop/common/circular_stack.h"

namespace snake {
    class PPO : public rlop::PPO {
    public:
        PPO(
            Int num_envs,
            bool render,
            Int num_steps,
            Int batch_size,
            Int num_epochs,
            double lr,
            double gamma,
            double clip_range,
            double clip_range_vf,
            bool normalize_advantage,
            double ent_coef,
            double vf_coef,
            double gae_lambda,
            double max_grad_norm,
            double target_kl,
            std::string output_path,
            torch::Device device
        ) :
            rlop::PPO(
                batch_size,
                num_epochs,
                lr,
                gamma,
                clip_range, 
                clip_range_vf, 
                normalize_advantage, 
                ent_coef, 
                vf_coef, 
                gae_lambda,
                max_grad_norm,
                target_kl,
                output_path,
                device
            ),
            problem_(num_envs, render),
            num_steps_(num_steps),
            score_stack_(problem_.max_num_steps())
        {}

        void Reset() override {
            rlop::PPO::Reset();
            for (Int env_i=0; env_i<problem_.num_problems(); ++env_i) {
                problem_.Reset(env_i, env_i);
            }
        }

        void RegisterLogItems() override {
            rlop::PPO::RegisterLogItems();
            log_items_["score"] = torch::Tensor();
            score_stack_.Reset();
        }
        
        std::shared_ptr<rlop::RolloutBuffer> MakeRolloutBuffer() const override {
            return std::make_shared<rlop::RolloutBuffer>(
                    num_steps_, 
                    problem_.num_problems(), 
                    problem_.observation_sizes(),
                    problem_.action_sizes()
                );
        }

        std::shared_ptr<rlop::PPOPolicy> MakePolicy() const override {
            return std::make_shared<PPOPolicy>(rollout_buffer_->observation_sizes(), problem_.NumActions());
        }

        Int NumEnvs() const override {
            return problem_.num_problems();
        }

        torch::Tensor ResetEnv() override {
            std::vector<torch::Tensor> observation_list(problem_.num_problems());
            #pragma omp parallel for
            for (Int env_i=0; env_i<problem_.num_problems(); ++env_i) {
                problem_.Reset(env_i);
                observation_list[env_i] = problem_.GetObservation(env_i);
            }
            problem_.Render();
            return torch::stack(observation_list);
        }

        std::array<torch::Tensor, 5> Step(const torch::Tensor& action) override {
            std::vector<torch::Tensor> observation_list(problem_.num_problems());
            std::vector<torch::Tensor> reward_list(problem_.num_problems());
            std::vector<torch::Tensor> terminated_list(problem_.num_problems());
            std::vector<torch::Tensor> score_list(problem_.num_problems());
            std::vector<torch::Tensor> terminal_obseravtion_list(problem_.num_problems());
            #pragma omp parallel for
            for (Int i=0; i<problem_.num_problems(); ++i) {
                Int num_foods = problem_.engines()[i].snakes()[0].num_foods;
                if (!problem_.Step(i, { problem_.GetAction(action[i].item<Int>()) })) {
                    double reward;
                    if (problem_.engines()[i].snakes()[0].alive)
                        reward = problem_.engines()[i].snakes()[0].num_foods - num_foods + problem_.engines()[i].min_num_foods() + 0.001 * problem_.engines()[i].num_steps();
                    else
                        reward = 0;
                    reward_list[i] = torch::tensor(reward, torch::kFloat32);
                    terminated_list[i] = torch::tensor(1, torch::kFloat32);
                    terminal_obseravtion_list[i] = problem_.GetObservation(i);
                    problem_.Reset(i);
                }
                else {
                    double reward = problem_.engines()[i].snakes()[0].num_foods - num_foods + 0.001;
                    reward_list[i] = torch::tensor(reward, torch::kFloat32);
                    terminated_list[i] = torch::tensor(0, torch::kFloat32);
                    terminal_obseravtion_list[i] = torch::zeros_like(problem_.GetObservation(i));
                }
                observation_list[i] = problem_.GetObservation(i);
                score_list[i] = torch::tensor(problem_.engines()[i].snakes()[0].num_foods, torch::kFloat32);
            }
            torch::Tensor next_observation = torch::stack(observation_list);
            torch::Tensor reward = torch::stack(reward_list);
            torch::Tensor terminated = torch::stack(terminated_list);
            torch::Tensor truncated = torch::zeros_like(terminated);
            torch::Tensor terminal_observation = torch::stack(terminal_obseravtion_list);
            score_stack_.PushBack(torch::stack(score_list));
            if (score_stack_.full())
                log_items_["score"] = torch::stack(score_stack_.vec()).mean();
            problem_.Render();
            return { next_observation, reward, terminated, truncated, terminal_observation };
        }

    protected:
        VectorProblem problem_;
        Int num_steps_;
        rlop::CircularStack<torch::Tensor> score_stack_;
    };
} 