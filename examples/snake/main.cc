#include "dqn.h"
#include "ppo.h"
#include "mcts.h"
#include "rlop/rl/evaluator.h"

int main() {
    using namespace snake;
    
    // PPO solver(16);
    // solver.Reset();
    // solver.Learn(1e7, 1, 20);
    // solver.Load("ppo_20240207_115356_688128.pth");

    // DQN solver(16);
    // solver.Reset();
    // solver.Learn(1e7, 1e3, 1e5);
    // solver.Load("dqn_20240207_141112_9600016.pth");

    // rlop::RLEvaluator evaluator;
    // evaluator.Reset();
    // auto [mean_reward, std_reward] = evaluator.Evaluate(&solver, 1e3);
    // std::cout << mean_reward << std::endl;
    // std::cout << std_reward << std::endl;

    MCTS solver(4);
    solver.Reset();
    solver.Evaluate(true, 1e7);

    return 0;
}
