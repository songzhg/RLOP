#pragma once
#include "ppo_policy.h"
#include "rlop/rl/gym_envs.h"
#include "rlop/rl/ppo/ppo.h"

namespace continuous_lunar_lander {
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
            num_steps_(num_steps)
        {
            py::dict kwargs;
            if (render)
                kwargs["render_mode"]  = "human";
            env_ = rlop::GymVectorEnv("LunarLanderContinuous-v2", num_envs, "async", kwargs);
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
            return std::make_unique<PPOPolicy>(rollout_buffer_->observation_sizes()[0], rollout_buffer_->action_sizes()[0]);
        }

        Int NumEnvs() const override {
            return env_.num_envs();
        }

        torch::Tensor ResetEnv() override {
            auto [observations, info] = env_.Reset();
            return rlop::gym_utils::ArrayToTensor(py::cast<py::array>(observations));
        }

        std::array<torch::Tensor, 5> Step(const torch::Tensor& actions) override {
            auto [observations, rewards, terminations, truncations, infos] = env_.Step(rlop::gym_utils::TensorToArray(actions)); 
            torch::Tensor next_observations = rlop::gym_utils::ArrayToTensor(py::cast<py::array>(observations));
            torch::Tensor final_observations = torch::zeros_like(next_observations);
            if (infos.contains("final_observation")) {
                Int i=0;
                auto observation_array = infos["final_observation"];
                for (const auto& obs : observation_array) {
                    if (!obs.is_none()) 
                        final_observations[i] = rlop::gym_utils::ArrayToTensor(py::cast<py::array>(obs));
                    ++i;
                }
            }
            return { 
                std::move(next_observations), 
                std::move(rlop::gym_utils::ArrayToTensor(rewards)),
                std::move(rlop::gym_utils::ArrayToTensor(terminations)),
                std::move(rlop::gym_utils::ArrayToTensor(truncations)),
                std::move(final_observations)
            };
        }

    protected:
        rlop::GymVectorEnv env_;
        Int num_steps_;
    };
}