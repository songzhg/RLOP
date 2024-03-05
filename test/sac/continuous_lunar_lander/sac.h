#pragma once
#include "sac_policy.h"
#include "rlop/rl/gym_envs.h"
#include "rlop/rl/sac/sac.h"

namespace continuous_lunar_lander {
    class SAC : public rlop::SAC {
    public:
        SAC(
            Int num_envs,
            bool render,
            Int replay_buffer_capacity,
            Int learning_starts,
            Int batch_size,
            double lr,
            double tau,
            double gamma,
            double ent_coef,
            bool auto_ent_coef,
            const std::optional<double>& target_entropy,
            Int train_freq,
            Int gradient_steps,
            Int target_update_interval,
            std::string output_path,
            const torch::Device& device
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
            if (render)
                kwargs["render_mode"]  = "human";
            env_ = rlop::GymVectorEnv("LunarLanderContinuous-v2", num_envs, "async", kwargs);
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

        torch::Tensor SampleActions() override {
            return rlop::gym_utils::ArrayToTensor(env_.action_space().attr("sample")());
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
        Int replay_buffer_capacity_;
    };
}