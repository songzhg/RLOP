#pragma once
#include "local_search.h"
#include "tabu_tables.h"

namespace rlop { 
    // Template class for Tabu Search optimization algorithm, extending LocalSearch.
    // This class template specializes in solving optimization problems by iteratively exploring
    // the solution space and avoiding cycles or previously visited solutions using a tabu table.
    //
    // Template parameter:
    //   TCost: The cost type of a solution, defaulted to double. Must be an arithmetic type.
    template<typename TCost = double>
    class TabuSearch : public LocalSearch<Int, TCost> {
    public:
        // Constructs a TabuSearch object with a specified maximum number of iterations without improvement.
        //
        // Parameters:
        //   max_num_unimproved_iters: The maximum number of consecutive iterations without improvement
        //   before the search is terminated.
        TabuSearch(Int max_num_unimproved_iters) : max_num_unimproved_iters_(max_num_unimproved_iters) {} 

        virtual ~TabuSearch() = default;
        
        // Pure virtual function to check if a given neighbor is tabu.
        //
        // Parameters:
        //   neighbor_i: The index of the neighbor being checked.
        //
        // Return:
        //   bool: Return true if the neighbor is tabu.
        virtual bool IsTabu(Int neighbor_i) = 0;

        // Pure virtual function to returns the total number of neighbors available for exploration.
        virtual Int NumNeighbors() const = 0;

        // Pure virtual function to evaluates and returns the cost of a neighbor.
        //
        // Parameters:
        //   neighbor_i: The index of the neighbor being evaluated.
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

        // Selects the next neighbor to consider, avoiding tabu neighbors unless it improves the best known cost.
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