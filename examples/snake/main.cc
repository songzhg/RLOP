#include "dqn.h"
#include "ppo.h"
#include "mcts.h"
#include "rlop/rl/evaluator.h"
#include "rlop/common/timer.h"

int main(int argc, char *argv[]) {
    using namespace snake;

    rlop::Timer timer;

    Int num_cpu = 16;
    Int num_time_steps = 1e7;

    if (argc <= 1 || std::string(argv[1]) == "dqn") {
        std::cout << "DQN training..." << std::endl;
        DQN solver(num_cpu);
        solver.Reset();
        timer.Start();
        solver.Learn(num_time_steps, 1e3);
        timer.Stop();
        solver.Save("data/snake/rlop_dqn.pth");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1e3);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;
    }
    else if (std::string(argv[1]) == "ppo") {
        std::cout << "PPO training..." << std::endl;
        PPO solver(num_cpu);
        solver.Reset();
        timer.Start();
        solver.Learn(num_time_steps, 1);
        timer.Stop();
        solver.Save("data/snake/rlop_ppo.pth");

        rlop::RLEvaluator evaluator;
        evaluator.Reset();
        auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1e3);
        std::cout << mean_reward << " " << std_reward << " " << timer.duration() << std::endl;
    }
    else if (std::string(argv[1]) == "mcts") {
        MCTS solver(4);
        solver.Reset();
        solver.Evaluate(num_time_steps, true);
    }
    return 0;
}
