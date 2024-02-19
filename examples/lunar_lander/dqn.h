#pragma once
#include "dqn_policy.h"
#include "rlop/rl/gym_envs.h"
#include "rlop/rl/dqn/dqn.h"

namespace lunar_lander {
    class DQN : public rlop::DQN {
    public:
        DQN(
            Int num_envs = 1,
            bool render = false,
            Int replay_buffer_capacity = 50000,
            Int learning_starts = 100,
            Int batch_size = 128,
            double lr = 0.00063,
            double tau = 1.0,
            double gamma = 0.99,
            double max_grad_norm = 10,
            double exploration_fraction = 0.16,
            double initial_eps = 1.0,
            double final_eps = 0.1,
            Int train_freq = 4,
            Int gradient_steps = 1,
            Int target_update_interval = 250,
            std::string output_path = "data/lunar_lander/rlop_dqn",
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
            replay_buffer_capacity_(replay_buffer_capacity),
            exploration_fraction_(exploration_fraction),
            initial_eps_(initial_eps),
            final_eps_(final_eps)
        {
            py::dict kwargs;
            if (render)
                kwargs["render_mode"]  = "human";
            env_ = rlop::GymVectorEnv("LunarLander-v2", num_envs, "async", kwargs);
        } 

        void Reset() override {
            rlop::DQN::Reset();
            eps_ = initial_eps_;
        }

        std::unique_ptr<rlop::ReplayBuffer> MakeReplayBuffer() const override {
            return std::make_unique<rlop::ReplayBuffer>(
                replay_buffer_capacity_, 
                env_.num_envs(), 
                rlop::gym_utils::ArrayShapeToTensorSizes(env_.observation_shape()),
                rlop::gym_utils::ArrayShapeToTensorSizes(env_.action_shape()),
                rlop::gym_utils::ArrayDtypeToTensorDtype(env_.observation_dtype()),
                rlop::gym_utils::ArrayDtypeToTensorDtype(env_.action_dtype())
            );
        }

        std::unique_ptr<rlop::QNet> MakeQNet() const override {
            auto ret = std::make_unique<QNet>(replay_buffer_->observation_sizes()[0], py::cast<Int>(env_.single_action_space().attr("n")));
            ret->Reset();
            return ret;
        }

        torch::Tensor SampleAction() override {
            return rlop::gym_utils::ArrayToTensor(env_.action_space().attr("sample")());
        }

        Int NumEnvs() const override {
            return env_.num_envs();
        }

        torch::Tensor ResetEnv() override {
            auto [observation, info] = env_.Reset();
            return rlop::gym_utils::ArrayToTensor(py::cast<py::array>(observation));
        }

        std::array<torch::Tensor, 3> Step(const torch::Tensor& action) override {
            auto [observation, reward, terminated, truncated, info] = env_.Step(rlop::gym_utils::TensorToArray(action)); 
            torch::Tensor done = torch::logical_or(rlop::gym_utils::ArrayToTensor(terminated), rlop::gym_utils::ArrayToTensor(truncated)).to(torch::kFloat32);
            return { 
                std::move(rlop::gym_utils::ArrayToTensor(py::cast<py::array>(observation))), 
                std::move(rlop::gym_utils::ArrayToTensor(reward)),
                std::move(done)
            };
        }

        void Update() override {
            rlop::DQN::Update();
            if (eps_ > final_eps_)
                eps_ = std::max(final_eps_, initial_eps_ - (initial_eps_ - final_eps_) * time_steps_  / (max_time_steps_ * exploration_fraction_));
        }

    protected:
        rlop::GymVectorEnv env_;
        Int replay_buffer_capacity_;
        double exploration_fraction_;
        double initial_eps_;
        double final_eps_;
    };
}