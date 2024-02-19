#pragma once
#include "mcts.h"

namespace rlop {
    class PUCT : public MCTS {
    public:
        PUCT(double coef = std::sqrt(2)) : MCTS(coef) {} 

        virtual ~PUCT() = default;

        virtual double GetProb(Int child_i) = 0;

        virtual double TreePolicy(Int child_i) override {
            if (path_.back()->children[child_i] == nullptr)
                return std::numeric_limits<double>::lowest();
            return path_.back()->children[child_i]->mean_reward + 
                coef_ * GetProb(child_i) * std::sqrt((double)path_.back()->children[child_i]->num_visits) / (1.0 + path_.back()->num_visits);
        }
    };
}
