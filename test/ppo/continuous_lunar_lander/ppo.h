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

        std::shared_ptr<rlop::RolloutBuffer> MakeRolloutBuffer() const override {
            return std::make_shared<rlop::RolloutBuffer>(
                num_steps_, 
                env_.num_envs(),
                rlop::pybind11_utils::ArrayShapeToTensorSizes(env_.observation_shape()),
                rlop::pybind11_utils::ArrayShapeToTensorSizes(env_.action_shape()),
                rlop::pybind11_utils::ArrayDtypeToTensorDtype(env_.observation_dtype()),
                rlop::pybind11_utils::ArrayDtypeToTensorDtype(env_.action_dtype())
            );
        }

        std::shared_ptr<rlop::PPOPolicy> MakePolicy() const override {
            return std::make_shared<PPOPolicy>(rollout_buffer_->observation_sizes()[0], rollout_buffer_->action_sizes()[0]);
        }

        Int NumEnvs() const override {
            return env_.num_envs();
        }

        torch::Tensor ResetEnv() override {
            auto [observations, info] = env_.Reset();
            return rlop::pybind11_utils::ArrayToTensor(py::cast<py::array>(observations));
        }

        std::array<torch::Tensor, 5> Step(const torch::Tensor& actions) override {
            auto [observation_py, rewards_array, termination_array, truncation_array, infos] = env_.Step(rlop::pybind11_utils::TensorToArray(actions)); 
            torch::Tensor observations = rlop::pybind11_utils::ArrayToTensor(py::cast<py::array>(observation_py));
            torch::Tensor rewards = rlop::pybind11_utils::ArrayToTensor(rewards_array);
            torch::Tensor terminations = rlop::pybind11_utils::ArrayToTensor(termination_array);
            torch::Tensor truncations = rlop::pybind11_utils::ArrayToTensor(truncation_array);
            torch::Tensor final_observations = torch::zeros_like(observations);
            if (infos.contains("final_observation")) {
                int i=0;
                auto observation_array = infos["final_observation"];
                for (const auto& obs : observation_array) {
                    if (!obs.is_none()) 
                        final_observations[i].copy_(rlop::pybind11_utils::ArrayToTensor(py::cast<py::array>(obs)));
                    else if (terminations[i].item<bool>() || truncations[i].item<bool>())
                        throw;
                    ++i;
                }
            }
            return { 
                observations, 
                rewards,
                terminations,
                truncations,
                final_observations
            };
        }

        const rlop::GymVectorEnv& env() const {
            return env_;
        }

    protected:
        rlop::GymVectorEnv env_;
        Int num_steps_;
    };
}