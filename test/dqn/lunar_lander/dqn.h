#pragma once
#include "dqn_policy.h"
#include "rlop/rl/gym_envs.h"
#include "rlop/rl/dqn/dqn.h"

namespace lunar_lander {
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
            const torch::Device& device,
            uint64_t seed
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
            replay_buffer_capacity_(replay_buffer_capacity)
        {
            py::dict kwargs;
            if (render)
                kwargs["render_mode"]  = "human";
            env_ = rlop::GymVectorEnv("LunarLander-v2", num_envs, "sync", kwargs);
            env_.Seed(seed);
            env_.single_action_space().attr("seed")(seed);
            linear_fn_ = rlop::MakeLinearFn(initial_eps, final_eps, exploration_fraction);
        } 

        void Reset() override {
            rlop::DQN::Reset();
            eps_ = linear_fn_(time_steps_ / (double)max_time_steps_); 
        }

        std::shared_ptr<rlop::ReplayBuffer> MakeReplayBuffer() const override {
            return std::make_unique<rlop::ReplayBuffer>(
                replay_buffer_capacity_, 
                env_.num_envs(), 
                rlop::pybind11_utils::ArrayShapeToTensorSizes(env_.observation_shape()),
                rlop::pybind11_utils::ArrayShapeToTensorSizes(env_.action_shape()),
                rlop::pybind11_utils::ArrayDtypeToTensorDtype(env_.observation_dtype()),
                rlop::pybind11_utils::ArrayDtypeToTensorDtype(env_.action_dtype())
            );
        }

        std::shared_ptr<rlop::RLPolicy> MakePolicy() const override {
            return std::make_shared<DQNPolicy>(replay_buffer_->observation_sizes()[0], py::cast<Int>(env_.single_action_space().attr("n")));
        }

        Int NumEnvs() const override {
            return env_.num_envs();
        }

        torch::Tensor SampleActions() override {
            std::vector<torch::Tensor> actions(env_.num_envs());
            for (Int i=0; i<env_.num_envs(); ++i) {
                actions[i] = rlop::pybind11_utils::ArrayToTensor(env_.single_action_space().attr("sample")());
            }
            return torch::stack(actions);
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

        void OnCollectRolloutStep() override {
            rlop::DQN::OnCollectRolloutStep();
            eps_ = linear_fn_(time_steps_ / (double)max_time_steps_); 
        }

        const rlop::GymVectorEnv& env() const {
            return env_;
        }

    protected:
        rlop::GymVectorEnv env_;
        Int replay_buffer_capacity_;
        std::function<double(double)> linear_fn_;
    };
}