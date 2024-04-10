#include "ppo.h"
#include "rlop/rl/evaluator.h"
#include "rlop/common/timer.h"

int main(int argc, char *argv[]) {
    using namespace continuous_lunar_lander;
 
	py::scoped_interpreter guard{}; 

    rlop::Timer timer;

    Int num_cpu = 16;
    Int num_time_steps = 1e6;
    Int num_experiments = 50;
    std::string path = "data/ppo/continuous_lunar_lander/rlop";

    rlop::torch_utils::SetRandomSeed(0);

    std::ofstream out(path + "_eval.txt");
    for (Int i=0; i<num_experiments; ++i) {
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
            path + "_" + std::to_string(i), // output_path
            torch::kCUDA, // device
            i
        );
        solver.Reset();
        timer.Restart();
        solver.Learn(num_time_steps, 1);
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