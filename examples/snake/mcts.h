
#pragma once
#include "problems/snake/problem.h"
#include "rlop/mcts/mcts.h"

namespace snake {
    class MCTS : public rlop::MCTS {
    public:
        MCTS(Int max_depth = 200, double coef = std::sqrt(2)) : 
            max_depth_(max_depth),
            rlop::MCTS(coef)
        {}
        
        void Reset() override {
            rlop::MCTS::Reset();
            path_.reserve(max_depth_);
        }

        Int NumChildStates() const override {
            return problem_.NumActions();
        }

        bool IsExpanded(const Node& node) const override {
            return node.num_visits > 3 * NumChildStates() && node.num_children == NumChildStates(); 
        }

        void RevertState() override {
            engine_bk_.set_seed(rand_.Uniform(uint64_t(0), uint64_t(100)));
            problem_.Reset(engine_bk_);
            depth_ = 0;
        }

        bool Step(Int child_i) override {
            Int dir = problem_.GetAction(child_i);
            if (!problem_.Step({ dir }) || depth_ >= max_depth_)
               return false;
            ++depth_;
            return true;
        }

        double Reward() override {
            if (problem_.engine().snakes()[0].alive)
                return (problem_.engine().snakes()[0].num_foods + problem_.engine().min_num_foods()) / (double)problem_.grid_size() + 0.002 * depth_ / (double)max_depth_;
            else 
                return problem_.engine().snakes()[0].num_foods / (double)problem_.grid_size() + 0.001 * depth_ / (double)max_depth_;
        }

        template<typename TEngine>
        Int NewSearch(TEngine&& engine, Int max_num_iters) {
            if (!engine.snakes()[0].alive)
                return kIntNull;
            Reset();
            engine_bk_ = std::forward<TEngine>(engine);
            Search(max_num_iters);
            Int best_i = kIntNull;
            double best_score = std::numeric_limits<double>::lowest();
            for (Int i=0; i < path_.front()->children.size(); ++i) {
                Int dir = problem_.GetAction(i);
                if(engine.Lookahead(0, dir)) {
                    double score = path_.front()->children[i]->num_visits;
                    if (score > best_score) {
                        best_score = score;
                        best_i = i;
                    } 
                }
            } 
            return best_i;
        }

        void Evaluate(Int num_time_steps, bool render, Int max_num_iters = 30000) {
            Problem problem(render);
            problem.Reset();
            for (Int i=0; i< num_time_steps; ++i) {
                Int dir = NewSearch(problem.engine(), max_num_iters);
                if (!problem.Step({ dir == kIntNull? 0 : dir }))
                    problem.Reset();
                problem.Render();
            }
        }

    private:
        Problem problem_;
        Int max_depth_;
        Engine engine_bk_; 
        Int depth_;
    };
}