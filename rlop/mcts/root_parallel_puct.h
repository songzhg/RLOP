#pragma once
#include "root_parallel_mcts.h"

namespace rlop {
    class RootParallelPUCT : public RootParallelMCTS {
    public:
        RootParallelPUCT(Int num_envs, double coef = std::sqrt(2)) : RootParallelMCTS(num_envs, coef) {} 

        virtual ~RootParallelPUCT() = default;

        // Pure virtual function to retrieve the probability of choosing a particular child node as indicated
        // by a policy for a specified environment. This is a pure virtual function that should be implemented
        // to return the probability of selecting the child at index `child_i` according to some policy, 
        // typically provided by a neural network or other predictive model.
        //
        // Parameters:
        //   env_i: The index of environment.
        //   child_i: The index of the child node for which the selection probability is requested.
        //
        // Returns:
        //   double: The probability of selecting the child node at index `child_i`.
        virtual double GetProb(Int env_i, Int child_i) = 0;

        virtual double TreePolicy(Int env_i, Int child_i) override {
            if (paths_[env_i].back()->children[child_i] == nullptr)
                return std::numeric_limits<double>::lowest();
            return paths_[env_i].back()->children[child_i]->mean_reward + 
                coef_ * GetProb(env_i, child_i) * std::sqrt((double)paths_[env_i].back()->children[child_i]->num_visits) / (1.0 + paths_[env_i].back()->num_visits);
        }
    };
}
