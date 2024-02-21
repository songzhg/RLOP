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

    if (argc <= 1 || std::string(argv[1]) == "ppo") {
        PPO solver(num_cpu);
        solver.Reset();
        timer.Start();
        solver.Learn(num_time_steps, 1);
        timer.Stop();
        solver.Save("data/continuous_lunar_lander/rlop_ppo.pth");
        std::ofstream out("data/continuous_lunar_lander/rlop_ppo_eval.txt");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1000);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;
        out << mean_reward << "\t" << std_reward << "\t" << timer.duration() << std::endl;
    }
    else if (std::string(argv[1]) == "sac") {
        SAC solver(num_cpu);
        solver.Reset();
        timer.Start();
        solver.Learn(num_time_steps, 1e3);
        timer.Stop();
        solver.Save("data/continuous_lunar_lander/rlop_sac.pth");
        std::ofstream out("data/continuous_lunar_lander/rlop_sac_eval.txt");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1000);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;
        out << mean_reward << "\t" << std_reward << "\t" << timer.duration() << std::endl;
    }
    return 0;
}