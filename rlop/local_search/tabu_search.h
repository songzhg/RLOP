#pragma once
#include "local_search.h"
#include "tabu_tables.h"

namespace rlop { 
    template<typename TCost = double>
    class TabuSearch : public LocalSearch<Int, TCost> {
    public:
        TabuSearch(Int max_num_unimproved_iters) : max_num_unimproved_iters_(max_num_unimproved_iters) {} 

        virtual ~TabuSearch() = default;
        
        virtual bool IsTabu(Int neighbor_i) = 0;

        virtual Int NumNeighbors() const = 0;

        virtual TCost EvaluateNeighbor(Int neighbor_i) = 0;

        virtual void Reset() override {
            LocalSearch<Int, TCost>::Reset();
            num_unimproved_iters_ = 0; 
        }

        virtual bool Proceed() override {
            if (!LocalSearch<Int, TCost>::Proceed())
                return false;
            return num_unimproved_iters_ < max_num_unimproved_iters_;
        }

        virtual std::optional<Int> Select() override {
            Int best = kIntNull;
            double best_cost = std::numeric_limits<double>::max();
            for (Int i=0; i<NumNeighbors(); ++i) {
                double cost = EvaluateNeighbor(i);
                if (cost >= this->best_cost_ && IsTabu(i))
                    continue;
                if (cost < best_cost) {
                    best = i;
                    best_cost = cost;
                }
            }
            if (best == kIntNull)
                return std::nullopt;
            return { best };
        }

        virtual void Improved() override {
            LocalSearch<Int, TCost>::Improved();
            num_unimproved_iters_ = 0;    
        }

        virtual void Unimproved() override {
            ++num_unimproved_iters_;
        }

        Int num_unimproved_iters() const {
            return num_unimproved_iters_;
        }

        Int max_num_unimproved_iters() const {
            return max_num_unimproved_iters_;
        }

    protected:
        Int num_unimproved_iters_ = 0;
        Int max_num_unimproved_iters_;
    };
}