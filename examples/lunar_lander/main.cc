#include "dqn.h"
#include "ppo.h"
#include "rlop/rl/evaluator.h"
#include "rlop/common/timer.h"

int main() {
    using namespace lunar_lander;
 
	py::scoped_interpreter guard{}; 

    rlop::Timer timer;

    Int num_cpu = 16;
    Int num_time_steps = 1e7;

    // DQN solver(num_cpu);
    // solver.Reset();
    // timer.Start();
    // solver.Learn(num_time_steps, 1e3);
    // timer.Stop();
    // solver.Save("data/lunar_lander/rlop_dqn.pth");
    // std::ofstream out("data/lunar_lander/rlop_dqn_eval.txt");

    PPO solver(num_cpu);
    solver.Reset();
    timer.Start();
    solver.Learn(num_time_steps, 1);
    timer.Stop();
    solver.Save("data/lunar_lander/rlop_ppo.pth");
    std::ofstream out("data/lunar_lander/rlop_ppo_eval.txt");

    rlop::RLEvaluator evaluator;
    evaluator.Reset();
    auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1000);
    std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;
    out << mean_reward << "\t" << std_reward << "\t" << timer.duration() << std::endl;
}