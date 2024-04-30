#include "sac.h"
#include "rlop/rl/evaluator.h"
#include "rlop/common/timer.h"

int main(int argc, char *argv[]) {
    using namespace continuous_lunar_lander;
 
	py::scoped_interpreter guard{};

    rlop::Timer timer;

    Int num_cpu = 16;
    Int num_time_steps = 1e6;
    Int num_experiments = 5;
    std::string path = "data/sac/continuous_lunar_lander/rlop";

    rlop::torch_utils::SetRandomSeed(0);

    std::ofstream out(path + "_eval.txt");
    for (Int i=0; i<num_experiments; ++i) {
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
            path + "_" + std::to_string(i), // output_path
            torch::kCUDA, // device
            i
        );
        solver.Reset();
        timer.Restart();
        solver.Learn(num_time_steps, 1e2);
        timer.Stop();
        // solver.Save(path + "_" + std::to_string(i) + ".pth");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 100);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() / 1000.0 << std::endl;
        out << mean_reward << "\t" << std_reward << "\t" << timer.duration() / 1000.0 << std::endl; 
    }
    
    return 0;
}