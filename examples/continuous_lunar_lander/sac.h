#pragma once
#include "sac_policy.h"
#include "rlop/rl/gym_envs.h"
#include "rlop/rl/sac/sac.h"

namespace continuous_lunar_lander {
    class SAC : public rlop::SAC {
    public:
        SAC(
            Int num_envs = 1,
            bool render = false,
            Int replay_buffer_capacity = 50000,
            Int learning_starts = 100,
            Int batch_size = 256,
            double lr = 3e-4,
            double tau = 0.01,
            double gamma = 0.99,
            double ent_coef = 1.0,
            bool auto_ent_coef = true,
            const std::optional<double>& target_entropy = std::nullopt,
            Int train_freq = 1,
            Int gradient_steps = 1,
            Int target_update_interval = 1,
            std::string output_path = "data/continuous_lunar_lander/rlop_sac",
            const torch::Device& device = torch::kCUDA
        ) :
            rlop::SAC(
                learning_starts,
                batch_size,
                lr,
                tau,
                gamma,
                ent_coef,
                auto_ent_coef,
                target_entropy,
                train_freq,
                gradient_steps,
                target_update_interval,
                output_path,
                device
            ),
            replay_buffer_capacity_(replay_buffer_capacity)
        {
            py::dict kwargs;
            kwargs["continuous"]  = true;
            if (render)
                kwargs["render_mode"]  = "human";
            env_ = rlop::GymVectorEnv("LunarLander-v2", num_envs, "async", kwargs);
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

        std::unique_ptr<rlop::SACActor> MakeActor() const override {
            return std::make_unique<SACActor>(replay_buffer_->observation_sizes()[0], replay_buffer_->action_sizes()[0]);
        }

        std::unique_ptr<rlop::SACCritic> MakeCritic() const override {
            return std::make_unique<SACCritic>(2, replay_buffer_->observation_sizes()[0], replay_buffer_->action_sizes()[0]);
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

    protected:
        rlop::GymVectorEnv env_;
        Int replay_buffer_capacity_;
    };
}