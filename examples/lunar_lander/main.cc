#include "dqn.h"
#include "ppo.h"
#include "rlop/rl/evaluator.h"
#include "rlop/common/timer.h"

int main(int argc, char *argv[]) {
    using namespace lunar_lander;
 
	py::scoped_interpreter guard{}; 

    rlop::Timer timer;

    Int num_cpu = 16;
    Int num_time_steps = 1e7;
    std::string path = "data/lunar_lander/rlop";

    if (argc <= 1 || std::string(argv[1]) == "dqn") {
        std::cout << "DQN training..." << std::endl;
        DQN solver(
            num_cpu, // num_envs
            false, // render
            50000, // replay_buffer_capacity
            100, // learning_starts
            128, // batch_size
            6.3e-4, // lr
            1.0, // tau
            0.99, // gamma
            10, // max_grad_norm
            0.12, // exploration_fraction
            1.0, // initial_eps
            0.1, // final_eps
            4, // train_freq
            1, // gradient_steps
            250, // target_update_interval
            path + "_dqn", // output_path
            torch::kCUDA // device
        );
        solver.Reset();
        timer.Start();
        solver.Learn(num_time_steps, 1e3);
        timer.Stop();
        solver.Save(path + "_dqn.pth");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1000);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;

    }
    else if (std::string(argv[1]) == "ppo") {
        std::cout << "PPO training..." << std::endl;
        PPO solver(
            num_cpu, // num_envs
            false, // render
            1024, // num_steps
            64, // batch_size
            4, // num_epochs
            3e-4, // lr
            0.99, // gamma
            0.2, // clip_range
            0, // clip_range_vf
            false, // normalize_advantage
            0.01, // ent_coef
            0.1, // vf_coef
            0.98, // gae_lambda
            0.5, // max_grad_norm
            0, // target_kl
            path + "_ppo", // output_path
            torch::kCUDA // device
        );
        solver.Reset();
        timer.Start();
        solver.Learn(num_time_steps, 1);
        timer.Stop();
        solver.Save(path + "_ppo.pth");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1000);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;
    }
    return 0;
}