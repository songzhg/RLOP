#pragma once
#include "ppo_policy.h"
#include "rlop/rl/gym_envs.h"
#include "rlop/rl/ppo/ppo.h"

namespace lunar_lander {
    class PPO : public rlop::PPO {
    public:
        PPO(
            Int num_envs = 1,
            bool render = false,
            Int num_steps = 1024,
            Int batch_size = 64,
            Int num_epochs = 4,
            double lr = 3e-4,
            double gamma = 0.99,
            double clip_range = 0.2,
            double clip_range_vf = 0,
            bool normalize_advantage = false,
            double ent_coef = 0.01,
            double vf_coef = 0.5,
            double gae_lambda = 0.98,
            double max_grad_norm = 0.5,
            double target_kl = 0,
            std::string output_path = "data/lunar_lander/rlop_ppo",
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
            num_steps_(num_steps)
        {
            py::dict kwargs;
            if (render)
                kwargs["render_mode"]  = "human";
            env_ = rlop::GymVectorEnv("LunarLander-v2", num_envs, "async", kwargs);
        }

        std::unique_ptr<rlop::RolloutBuffer> MakeRolloutBuffer() const override {
            return std::make_unique<rlop::RolloutBuffer>(
                num_steps_, 
                env_.num_envs(),
                rlop::gym_utils::ArrayShapeToTensorSizes(env_.observation_shape()),
                rlop::gym_utils::ArrayShapeToTensorSizes(env_.action_shape()),
                rlop::gym_utils::ArrayDtypeToTensorDtype(env_.observation_dtype()),
                rlop::gym_utils::ArrayDtypeToTensorDtype(env_.action_dtype())
            );
        }

        std::unique_ptr<rlop::PPOPolicy> MakePPOPolicy() const override {
            auto ret = std::make_unique<PPOPolicy>(rollout_buffer_->observation_sizes()[0], py::cast<Int>(env_.single_action_space().attr("n")));
            ret->Reset();
            return ret;
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
        Int num_steps_;
    };
}