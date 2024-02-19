#pragma once
#include "problems/snake/problem.h"
#include "ppo_policy.h"
#include "rlop/rl/ppo/ppo.h"
#include "rlop/common/circular_stack.h"

namespace snake {
    class PPO : public rlop::PPO {
    public:
        PPO(
            Int num_envs = 1,
            bool render = false,
            Int num_steps = 2048,
            Int batch_size = 64,
            Int num_epochs = 4,
            double lr = 1e-4,
            double gamma = 0.99,
            double clip_range = 0.2,
            double clip_range_vf = 0,
            bool normalize_advantage = true,
            double ent_coef = 0.01,
            double vf_coef = 0.1,
            double gae_lambda = 0.95,
            double max_grad_norm = 10,
            double target_kl = 0.1,
            std::string output_path = "./ppo",
            torch::Device device = torch::kCUDA
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
        
        std::unique_ptr<rlop::RolloutBuffer> MakeRolloutBuffer() const override {
            return std::make_unique<rlop::RolloutBuffer>(
                    num_steps_, 
                    problem_.num_problems(), 
                    problem_.observation_sizes(),
                    problem_.action_sizes(),
                    torch::kFloat32,
                    torch::kInt64 
                );
        }

        std::unique_ptr<rlop::PPOPolicy> MakePPOPolicy() const override {
            auto ret = std::make_unique<PPOPolicy>(rollout_buffer_->observation_sizes(), problem_.NumActions());
            ret->Reset();
            return ret;
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

        std::array<torch::Tensor, 3> Step(const torch::Tensor& action) override {
            std::vector<torch::Tensor> observation_list(problem_.num_problems());
            std::vector<torch::Tensor> reward_list(problem_.num_problems());
            std::vector<torch::Tensor> done_list(problem_.num_problems());
            std::vector<torch::Tensor> score_list(problem_.num_problems());
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
                    done_list[i] = torch::tensor(1, torch::kFloat32);
                    problem_.Reset(i);
                }
                else {
                    double reward = problem_.engines()[i].snakes()[0].num_foods - num_foods + 0.001;
                    reward_list[i] = torch::tensor(reward, torch::kFloat32);
                    done_list[i] = torch::tensor(0, torch::kFloat32);
                }
                observation_list[i] = problem_.GetObservation(i);
                score_list[i] = torch::tensor(problem_.engines()[i].snakes()[0].num_foods, torch::kFloat32);
            }
            torch::Tensor next_observation = torch::stack(observation_list);
            torch::Tensor reward = torch::stack(reward_list);
            torch::Tensor done = torch::stack(done_list);
            score_stack_.PushBack(torch::stack(score_list));
            if (score_stack_.full())
                log_items_["score"] = torch::stack(score_stack_.vec()).mean();
            problem_.Render();
            return { next_observation, reward, done };
        }

    protected:
        VectorProblem problem_;
        Int num_steps_;
        rlop::CircularStack<torch::Tensor> score_stack_;
    };
} 