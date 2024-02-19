#pragma once
#include "rlop/common/base_algorithm.h"

namespace rlop { 
    template<typename TNeighbor, typename TCost = double>
    class LocalSearch : public BaseAlgorithm {
    public:
        static_assert(std::is_arithmetic_v<TCost>, "LocalSearch: cost type should be arithmetic"); 

        LocalSearch() = default;

        virtual ~LocalSearch() = default;

        virtual TCost EvaluateSolution() = 0;

        virtual void RecordSolution() = 0;

        virtual std::optional<TNeighbor> Select() = 0;

        virtual bool Step(const TNeighbor& neighbor) = 0;

        virtual void Reset() override {}
        
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
        
        virtual bool Proceed() {
            return num_iters_ < max_num_iters_;
        }

        virtual void Improved() {
            RecordSolution();
        }

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