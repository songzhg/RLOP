#include "ppo.h"
#include "sac.h"
#include "rlop/rl/evaluator.h"
#include "rlop/common/timer.h"

int main(int argc, char *argv[]) {
    using namespace continuous_lunar_lander;

    py::scoped_interpreter guard{}; 

    rlop::Timer timer;

    Int num_cpu = 16;
    Int num_time_steps = 1e7;
    std::string path = "data/continuous_lunar_lander/rlop";

    if (argc <= 1 || std::string(argv[1]) == "ppo") {
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
        solver.Save("data/continuous_lunar_lander/rlop_ppo.pth");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1000);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;
    }
    else if (std::string(argv[1]) == "sac") {
        std::cout << "SAC training..." << std::endl;
        SAC solver(
            num_cpu, // num_envs
            false, // render
            50000, // replay_buffer_capacity
            100, // learning_starts
            256, // batch_size
            3e-4, // lr
            0.01, // tau
            0.99, // gamma
            1.0, // ent_coef
            true, // auto_ent_coef
            std::nullopt, // target_entropy
            1, // train_freq
            1, // gradient_steps
            1, // target_update_interval
            path + "_sac", // output_path
            torch::kCUDA // device
        );
        solver.Reset();
        timer.Start();
        solver.Learn(num_time_steps, 1e3);
        timer.Stop();
        solver.Save("data/continuous_lunar_lander/rlop_sac.pth");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1000);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;
    }
    return 0;
}