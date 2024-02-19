#pragma once
#include "root_parallel_mcts.h"

namespace rlop {
    class RootParallelPUCT : public RootParallelMCTS {
    public:
        RootParallelPUCT(Int num_envs, double coef = std::sqrt(2)) : RootParallelMCTS(num_envs, coef) {} 

        virtual ~RootParallelPUCT() = default;

        virtual double GetProb(Int env_i, Int child_i) = 0;

        virtual double TreePolicy(Int env_i, Int child_i) override {
            if (paths_[env_i].back()->children[child_i] == nullptr)
                return std::numeric_limits<double>::lowest();
            return paths_[env_i].back()->children[child_i]->mean_reward + 
                coef_ * GetProb(env_i, child_i) * std::sqrt((double)paths_[env_i].back()->children[child_i]->num_visits) / (1.0 + paths_[env_i].back()->num_visits);
        }
    };
}
