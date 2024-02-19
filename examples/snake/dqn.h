#pragma once
#include "problems/snake/problem.h"
#include "dqn_policy.h"
#include "rlop/rl/dqn/dqn.h"
#include "rlop/common/circular_stack.h"

namespace snake {
    class DQN : public rlop::DQN {
    public:
        DQN(
            Int num_envs = 1,
            bool render = false,
            Int replay_buffer_capacity = 1e5,
            Int learning_starts = 1e3,
            Int batch_size = 32,
            double lr = 1e-4,
            double tau = 1.0,
            double gamma = 0.99,
            double max_grad_norm = 10,
            double exploration_fraction = 0.1,
            double initial_eps = 1.0,
            double final_eps = 0.05,
            Int train_freq = 1,
            Int gradient_steps = 1,
            Int target_update_interval = 1e4,
            std::string output_path = "./dqn",
            const torch::Device& device = torch::kCUDA
        ) :
            rlop::DQN(
                learning_starts,
                batch_size,
                lr,
                tau,
                gamma,
                initial_eps,
                max_grad_norm,
                train_freq,
                gradient_steps,
                target_update_interval,
                output_path,
                device
            ),
            problem_(num_envs, render),
            replay_buffer_capacity_(replay_buffer_capacity),
            exploration_fraction_(exploration_fraction),
            initial_eps_(initial_eps),
            final_eps_(final_eps),
            score_stack_(problem_.max_num_steps())
        {}

        void Reset() override {
            rlop::DQN::Reset();
            for (Int env_i=0; env_i<problem_.num_problems(); ++env_i) {
                problem_.Reset(env_i, env_i);
            }
            eps_ = initial_eps_;
        }

        void RegisterLogItems() override {
            rlop::DQN::RegisterLogItems();
            log_items_["score"] = torch::Tensor();
            score_stack_.Reset();
        }

        std::unique_ptr<rlop::ReplayBuffer> MakeReplayBuffer() const override {
            return std::make_unique<rlop::ReplayBuffer>(
                    replay_buffer_capacity_, 
                    problem_.num_problems(), 
                    problem_.observation_sizes(),
                    problem_.action_sizes(),
                    torch::kFloat32,
                    torch::kInt64 
                );
        }

        std::unique_ptr<rlop::QNet> MakeQNet() const override {
            auto ret = std::make_unique<QNet>(replay_buffer_->observation_sizes(), problem_.NumActions());
            ret->Reset();
            return ret;
        }

        torch::Tensor SampleAction() override {
            return torch::randint(0, problem_.NumActions(), { problem_.num_problems() }, torch::TensorOptions().device(device_).dtype(torch::kInt64));
        }

        Int NumEnvs() const override {
            return problem_.num_problems();
        }

        torch::Tensor ResetEnv() override {
            std::vector<torch::Tensor> observation_list(problem_.num_problems());
            #pragma omp parallel for
            for (Int i=0; i<problem_.num_problems(); ++i) {
                problem_.Reset(i);
                observation_list[i] = problem_.GetObservation(i);
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

        void Update() override {
            rlop::DQN::Update();
            if (eps_ > final_eps_)
                eps_ = std::max(0.0, eps_ - (initial_eps_ - final_eps_)  / (max_time_steps_ / problem_.num_problems() * exploration_fraction_));
        }

    protected:
        VectorProblem problem_;
        Int replay_buffer_capacity_;
        double exploration_fraction_;
        double initial_eps_;
        double final_eps_;
        rlop::CircularStack<torch::Tensor> score_stack_;
    };
}