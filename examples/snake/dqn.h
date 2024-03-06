#pragma once
#include "problems/snake/problem.h"
#include "dqn_policy.h"
#include "rlop/rl/dqn/dqn.h"
#include "rlop/common/circular_stack.h"

namespace snake {
    class DQN : public rlop::DQN {
    public:
        DQN(
            Int num_envs,
            bool render,
            Int replay_buffer_capacity,
            Int learning_starts,
            Int batch_size,
            double lr,
            double tau,
            double gamma,
            double max_grad_norm,
            double exploration_fraction,
            double initial_eps,
            double final_eps,
            Int train_freq,
            Int gradient_steps,
            Int target_update_interval,
            std::string output_path,
            const torch::Device& device
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
            score_stack_(problem_.max_num_steps())
        {
            linear_fn_ = rlop::MakeLinearFn(initial_eps, final_eps, exploration_fraction);
        }

        void Reset() override {
            rlop::DQN::Reset();
            for (Int env_i=0; env_i<problem_.num_problems(); ++env_i) {
                problem_.Reset(env_i, env_i);
            }
            eps_ = linear_fn_(time_steps_ / (double)max_time_steps_); 
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
            return std::make_unique<QNet>(replay_buffer_->observation_sizes(), problem_.NumActions());
        }

        torch::Tensor SampleActions() override {
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

        std::array<torch::Tensor, 5> Step(const torch::Tensor& actions) override {
            std::vector<torch::Tensor> observation_list(problem_.num_problems());
            std::vector<torch::Tensor> reward_list(problem_.num_problems());
            std::vector<torch::Tensor> termination_list(problem_.num_problems());
            std::vector<torch::Tensor> score_list(problem_.num_problems());
            std::vector<torch::Tensor> terminal_obseravtion_list(problem_.num_problems());
            #pragma omp parallel for
            for (Int i=0; i<problem_.num_problems(); ++i) {
                Int num_foods = problem_.engines()[i].snakes()[0].num_foods;
                if (!problem_.Step(i, { problem_.GetAction(actions[i].item<Int>()) })) {
                    double reward;
                    if (problem_.engines()[i].snakes()[0].alive)
                        reward = problem_.engines()[i].snakes()[0].num_foods - num_foods + problem_.engines()[i].min_num_foods() + 0.001 * problem_.engines()[i].num_steps();
                    else
                        reward = 0;
                    reward_list[i] = torch::tensor(reward, torch::kFloat32);
                    termination_list[i] = torch::tensor(1, torch::kFloat32);
                    terminal_obseravtion_list[i] = problem_.GetObservation(i);
                    problem_.Reset(i);
                }
                else {
                    double reward = problem_.engines()[i].snakes()[0].num_foods - num_foods + 0.001;
                    reward_list[i] = torch::tensor(reward, torch::kFloat32);
                    termination_list[i] = torch::tensor(0, torch::kFloat32);
                    terminal_obseravtion_list[i] = torch::zeros_like(problem_.GetObservation(i));
                }
                observation_list[i] = problem_.GetObservation(i);
                score_list[i] = torch::tensor(problem_.engines()[i].snakes()[0].num_foods, torch::kFloat32);
            }
            torch::Tensor next_observations = torch::stack(observation_list);
            torch::Tensor rewards = torch::stack(reward_list);
            torch::Tensor terminations = torch::stack(termination_list);
            torch::Tensor truncations = torch::zeros_like(terminations);
            torch::Tensor terminal_observations = torch::stack(terminal_obseravtion_list);
            score_stack_.PushBack(torch::stack(score_list));
            if (score_stack_.full())
                log_items_["score"] = torch::stack(score_stack_.vec()).mean();
            problem_.Render();
            return { next_observations, rewards, terminations, truncations, terminal_observations };
        }

        void Update() override {
            rlop::DQN::Update();
            eps_ = linear_fn_(time_steps_ / (double)max_time_steps_); 
        }

    protected:
        VectorProblem problem_;
        Int replay_buffer_capacity_;
        rlop::CircularStack<torch::Tensor> score_stack_;
        std::function<double(double)> linear_fn_;
    };
}