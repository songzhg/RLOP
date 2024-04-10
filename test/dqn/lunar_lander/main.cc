#include "dqn.h"
#include "rlop/rl/evaluator.h"
#include "rlop/common/timer.h"

int main(int argc, char *argv[]) {
    using namespace lunar_lander;
 
	py::scoped_interpreter guard{}; 

    rlop::Timer timer;

    Int num_cpu = 16;
    Int num_time_steps = 5e6;
    Int num_experiments = 50;
    std::string path = "data/dqn/lunar_lander/rlop";

    rlop::torch_utils::SetRandomSeed(0);

    std::ofstream out(path + "_eval.txt");
    for (Int i=0; i<num_experiments; ++i) {
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
            path + "_" + std::to_string(i), // output_path
            torch::kCUDA, // device
            i
        );
        solver.Reset();
        timer.Restart();
        solver.Learn(num_time_steps, 1e3);
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