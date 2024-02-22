#pragma once
#include "rlop/common/base_algorithm.h"

namespace rlop { 
    // A template class for implementing local search algorithms, where the goal is to find an
    // optimal or near-optimal solution by iteratively exploring neighboring solutions.
    //
    // Template parameters:
    //   TNeighbor: The type representing a neighbor solution.
    //   TCost: The cost type of a solution, defaulted to double. Must be an arithmetic type.
    template<typename TNeighbor, typename TCost = double>
    class LocalSearch : public BaseAlgorithm {
    public:
        static_assert(std::is_arithmetic_v<TCost>, "LocalSearch: cost type should be arithmetic"); 

        LocalSearch() = default;

        virtual ~LocalSearch() = default;

        // Pure virtual function to evaluate the cost of the current solution.
        virtual TCost EvaluateSolution() = 0;

        // Pure virtual function to record the current best solution.
        virtual void RecordSolution() = 0;

        // Pure virtual function to select and return a optional neighbor to consider next.
        //
        // Returns:
        //   std::optional<TNeighbor>: The neighbor being selected. If there is no legal neighbor, 
        //                             return std::nullopt.
        virtual std::optional<TNeighbor> Select() = 0;

        // Pure virtual function to perform a search step using the provided neighbor. 
        //
        // Parameters:
        //   neighbor: The neighbor to consider moving to.
        //
        // Returns:
        //   bool: Returns true if the search is going to continue. Returns false if the search is 
        //         going to stop.
        virtual bool Step(const TNeighbor& neighbor) = 0;

        // Resets the algorithm.
        virtual void Reset() override {}

        // Executes the search algorithm for a maximum number of iterations.
        //
        // Parameters:
        //   max_num_iters: The maximum number of iterations to perform.
        virtual void Search(Int max_num_iters) {
            num_iters_= 0;
            max_num_iters_ = max_num_iters;
            best_cost_ = EvaluateSolution();
            RecordSolution();
            while (Proceed()) {
                auto neighbor = Select();
                if (!neighbor || !Step(*neighbor))
                    break;
                TCost cost = EvaluateSolution();
                if (cost < best_cost_) {
                    best_cost_ = cost;
                    Improved();
                }
                else
                    Unimproved();
                Update();
            }
        }
        
        // Checks if the search should continue.
        virtual bool Proceed() {
            return num_iters_ < max_num_iters_;
        }

        // Invoked when an improvement is found.
        virtual void Improved() {
            RecordSolution();
        }

        // Invoked when no improvement is found in the current step.
        virtual void Unimproved() {}

        virtual void Update() {
            ++num_iters_;
        }

        TCost best_cost() const {
            return best_cost_;
        }

        Int num_iters() const {
            return num_iters_;
        }

        Int max_num_iters() const {
            return max_num_iters_;
        }
        
    protected:
        TCost best_cost_ = std::numeric_limits<TCost>::max();
        Int num_iters_ = 0;
        Int max_num_iters_ = 0;
    };
}