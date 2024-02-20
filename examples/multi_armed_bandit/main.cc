#include "solvers.h"
#include "rlop/common/timer.h"

int main() {
    using namespace multi_armed_bandit;

    rlop::Timer timer;
    Int num_experiments = 2000;
    Int num_arms = 10;
    Int max_num_steps = 1000;
    
    std::vector<BaseSolver*> solvers;
    solvers.push_back(new EpsilonGreedySolver("epsilon_greedy", num_arms));
    solvers.push_back(new SoftmaxSolver("softmax", num_arms));
    solvers.push_back(new UCB1Solver("ucb1", num_arms));
    solvers.push_back(new PersuitSolver("persuit", num_arms));
    solvers.push_back(new PursuitEpsilonGreedySolver("persuit_epsilon_greedy", num_arms));

    std::vector<std::vector<Int>> average_rewards(solvers.size(), std::vector<Int>(max_num_steps, 0)); 
    std::vector<std::vector<Int>> total_num_opts(solvers.size(), std::vector<Int>(max_num_steps, 0));

    for (Int i=0; i<solvers.size(); ++i) {
        timer.Restart();
        for (Int j=0; j<num_experiments; ++j) {
            solvers[i]->Reset();
            solvers[i]->Solve(max_num_steps);
            for (Int step=0; step<max_num_steps; ++step) {
                average_rewards[i][step] += (solvers[i]->total_rewards()[step] / (1 + step));
                total_num_opts[i][step] += solvers[i]->num_opts()[step];
            }
        }
        timer.Stop(); 
        std::cout << solvers[i]->name() + ": " << timer.duration() << "ms" << std::endl;
    }

    std::ofstream out("reward_results.txt");
    out << "num_steps";
    for (Int i=0; i<solvers.size(); ++i) {
        out << "\t" << solvers[i]->name();
    }
    out << std::endl;
    for (Int i=0; i<max_num_steps; ++i) {
        out << i;
        for (Int j=0; j<solvers.size(); ++j) {
            out << "\t" << average_rewards[j][i] / (double)num_experiments;
        }
        out << std::endl;
    }
    
    out = std::ofstream("opt_results.txt");
    out << "num_steps";
    for (Int i=0; i<solvers.size(); ++i) {
        out << "\t" << solvers[i]->name();
    }
    out << std::endl;
    for (Int i=0; i<max_num_steps; ++i) {
        out << i;
        for (Int j=0; j<solvers.size(); ++j) {
            out << "\t" << total_num_opts[j][i] / (double)num_experiments;
        }
        out << std::endl;
    }

    for (auto ptr : solvers) {
        delete ptr;
    }

    return 0;
}