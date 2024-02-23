#pragma once
#include "rlop/common/gym_utils.h"

namespace rlop {
    // GymEnv is a class that encapsulates an Gymnasium environment, allowing C++ code to interface
    // with Python-based Gym environments. For API of Gymnasium environments, please refer to 
    // https://gymnasium.farama.org/api/env/.
    class GymEnv {
    public:
        GymEnv() = default;

        GymEnv(
            const std::string& id,
            const py::kwargs& kwargs = py::dict(), 
            const py::object& max_episode_steps = py::none(), 
            const py::object& disable_env_checker = py::none(),
            const py::object& wrappers = py::none()
        ) {
            py::module gym = py::module_::import("gymnasium");
            env_ = gym.attr("make")(id, max_episode_steps, disable_env_checker, **kwargs);
            if (!wrappers.is_none()) {
                if (py::isinstance<py::list>(wrappers)) {
                    auto wrapper_list = py::cast<py::list>(wrappers);
                    for (const auto& wrapper : wrapper_list) {
                        env_ = wrapper(env_);
                    }
                }
                else
                    env_ = wrappers(env_);
            }
        }

        GymEnv(
            const py::object& env_spec,
            const py::kwargs& kwargs = py::dict(), 
            const py::object& max_episode_steps = py::none(), 
            const py::object& disable_env_checker = py::none(),
            const py::object& wrappers = py::none()
        ) {
            py::module gym = py::module_::import("gymnasium");
            env_ = gym.attr("make")(env_spec, max_episode_steps, disable_env_checker, **kwargs);
        }

        virtual ~GymEnv() = default;

        virtual std::tuple<py::object, py::dict> Reset(const py::object& seed = py::none(), const py::object& options = py::none()) {
            auto results = py::cast<py::tuple>(env_.attr("reset")(py::arg("seed")=seed, py::arg("options")=options));
            return { std::move(results[0]), std::move(py::cast<py::dict>(results[1])) };
        }

        virtual std::tuple<py::object, py::float_, bool, bool, py::dict> Step(Int env_i, const py::object& action) {
            auto results = py::cast<py::tuple>(env_.attr("step")(action));
            return {
                std::move(results[0]),
                py::cast<py::float_>(results[1]),
                py::cast<bool>(results[2]),
                py::cast<bool>(results[3]),
                std::move(py::cast<py::dict>(results[4])),
            };
        }

        virtual py::object Render() {
            return env_.attr("render")();
        }

        virtual void Close() {
            env_.attr("close")();
        }

        py::object observation_space() const {
            return env_.attr("observation_space");
        }

        py::object action_space() const {
            return env_.attr("action_space");
        }

        py::tuple observation_shape() const {
            return py::cast<py::tuple>(observation_space().attr("shape"));
        }

        py::tuple action_shape() const {
            return py::cast<py::tuple>(action_space().attr("shape"));
        }

        py::dtype observation_dtype() const {
            return observation_space().attr("dtype");
        }

        py::dtype action_dtype() const {
            return action_space().attr("dtype");
        }

        const py::object& env() const {
            return env_;
        }
        
    private:
        py::object env_;
    };

    // GymEnv is a class that encapsulates an vectorized Gymnasium environment, allowing C++ code to interface
    // with Python-based Gym environments. For API of vectorized Gymnasium environments, please refer to 
    // https://gymnasium.farama.org/api/vector/.
    class GymVectorEnv {
    public:
        GymVectorEnv() = default;

        GymVectorEnv(
            const std::string& id, 
            Int num_envs, 
            const std::string& vectorization_mode = "sync", 
            const py::kwargs& kwargs = py::dict(), 
            const py::dict& vector_kwargs = py::dict(),  
            const py::object& wrappers = py::none()  
        ) : num_envs_(num_envs) {
            py::module gym = py::module_::import("gymnasium");
            env_ = gym.attr("make_vec")(id, num_envs, vectorization_mode, vector_kwargs, wrappers, **kwargs);
        }

        GymVectorEnv(
            const py::object& env_spec, 
            Int num_envs, 
            const std::string& vectorization_mode = "sync", 
            const py::kwargs& kwargs = py::dict(), 
            const py::dict& vector_kwargs = py::dict(),  
            const py::object& wrappers = py::none()  
        ) : num_envs_(num_envs) {
            py::module gym = py::module_::import("gymnasium");
            env_ = gym.attr("make_vec")(env_spec, num_envs, vectorization_mode, vector_kwargs, wrappers, **kwargs);
        }
        
        virtual ~GymVectorEnv() = default;

        virtual std::tuple<py::object, py::dict> Reset(const py::object& seed = py::none(), const py::object& options = py::none()) {
            auto results = py::cast<py::tuple>(env_.attr("reset")(py::arg("seed")=seed, py::arg("options")=options));
            return { 
                std::move(results[0]),
                std::move(py::cast<py::dict>(results[1])),
            };
        }

        virtual std::tuple<py::object, py::array, py::array, py::array, py::dict> Step(const py::object& action) {
            auto results = py::cast<py::tuple>(env_.attr("step")(action));
            return {
                std::move(results[0]),
                std::move(py::cast<py::array>(results[1])),
                std::move(py::cast<py::array>(results[2])),
                std::move(py::cast<py::array>(results[3])),
                std::move(py::cast<py::dict>(results[4])),
            };
        }

        virtual void Close() {
            env_.attr("close")();
        }

        Int num_envs() const {
            return num_envs_;
        }

        py::object observation_space() const {
            return env_.attr("observation_space");
        }

        py::object action_space() const {
            return env_.attr("action_space");
        }

        py::object single_observation_space() const {
            return env_.attr("single_observation_space");
        }

        py::object single_action_space() const {
            return env_.attr("single_action_space");
        }

        py::tuple observation_shape() const {
            return py::cast<py::tuple>(single_observation_space().attr("shape"));
        }

        py::tuple action_shape() const {
            return py::cast<py::tuple>(single_action_space().attr("shape"));
        }

        py::dtype observation_dtype() const {
            return single_observation_space().attr("dtype");
        }

        py::dtype action_dtype() const {
            return single_action_space().attr("dtype");
        }

        const py::object& env() const {
            return env_;
        }

    private:
        py::object env_;
        Int num_envs_;
    };
}