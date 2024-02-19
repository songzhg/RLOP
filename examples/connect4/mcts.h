#pragma once
#include "problems/connect4/problem.h"
#include "rlop/mcts/root_parallel_puct.h"

namespace connect4 {
    class MCTS : public rlop::RootParallelMCTS {
    public:
        MCTS(double coef = std::sqrt(2)) : problem_(Board::kWidth_), rlop::RootParallelMCTS(Board::kWidth_, coef) {}

        void Reset() override {
            rlop::RootParallelMCTS::Reset();
            stacks_ = std::vector<std::vector<Int>>(problem_.num_problems());
            for (Int i=0; i<problem_.num_problems(); ++i) {
                paths_[i].reserve(64);
                stacks_[i].reserve(64);
            }
        }

        Int NumChildStates(Int env_i) const override {
            return problem_.NumMoves();
        }

        bool IsExpanded(Int env_i, const Node& node) const override {
            return node.num_visits > 3*NumChildStates(env_i) && node.num_children == NumChildStates(env_i); 
        }

        void RevertState(Int env_i) override {
            while (!stacks_[env_i].empty()) {
                problem_.Undo(env_i, stacks_[env_i].back());
                stacks_[env_i].pop_back();
            }
        }

        bool Step(Int env_i, Int child_i) override {
            Int move = problem_.GetMove(child_i);
            if (!problem_.Step(env_i, move))
                return false;
            stacks_[env_i].push_back(move);
            if (problem_.boards()[env_i].IsOver())
                return false;
            return true;
        }

        double Reward(Int env_i) override {
            if (problem_.boards()[env_i].Win()) {
                if (stacks_[env_i].size() % 2 == 0)
                    return 1;
                else
                    return -1;
            }
            else if (problem_.boards()[env_i].IsFull()) {
                return 0;
            }
            else {
                if (stacks_[env_i].size() % 2 == 0)
                    return -1;
                else
                    return 1;
            }
        }

        void UpdateNode(Int env_i, double reward) const override {
            if (paths_[env_i].size() % 2 == 1)
               rlop::RootParallelMCTS::UpdateNode(env_i, reward);
            else
               rlop::RootParallelMCTS::UpdateNode(env_i, -reward);
        }
        
        Int NewSearch(const Board& board, Int max_num_iters = 6000) {
            if (board.IsOver())
                return kIntNull;
            Reset();
            Int best_i = kIntNull;
            double score = std::numeric_limits<double>::lowest();
            #pragma omp parallel for
            for (Int i=0; i < problem_.NumMoves(); ++i) {
                problem_.Reset(i, board);
                if(problem_.Step(i, problem_.GetMove(i))) {
                    if (problem_.boards()[i].Win()) {
                        #pragma omp critical
                        {
                            best_i = i;
                            score = std::numeric_limits<double>::max();
                        }
                    }
                    else if (problem_.boards()[i].IsFull()) {
                        #pragma omp critical
                        if (0 > score) {
                            best_i = i;
                            score = 0;
                        } 
                    }
                    else {
                        Search(i, max_num_iters);
                        #pragma omp critical
                        if (paths_[i].front()->mean_reward > score) {
                            best_i = i;
                            score = paths_[i].front()->mean_reward;
                        } 
                    }
                    
                }
            } 
            return best_i; 
        }

    protected:
        VectorProblem problem_;
        std::vector<std::vector<Int>> stacks_;
    };
}