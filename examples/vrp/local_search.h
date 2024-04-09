#pragma once
#include "problems/vrp/problem.h"
#include "rlop/local_search/tabu_search.h"
 
namespace vrp {
    class LocalSearch : public rlop::TabuSearch<Int> {
    public:
        LocalSearch(const std::function<Int(Int, Int)>& get_cost, Int max_num_unimproved_iters = 50 ) : 
            rlop::TabuSearch<Int>(max_num_unimproved_iters),
            operator_space_(routes_),
            cost_manager_(routes_, get_cost),
            problem_(&routes_, &operator_space_, { &cost_manager_ })
        {}

        ~LocalSearch() = default;

        void Reset(const Routes& routes) {
            rlop::TabuSearch<Int>::Reset();
            routes_ = routes;
            operator_space_.Reset();
            cost_manager_.Reset();
        }

        void Reset(Routes&& routes) {
            rlop::TabuSearch<Int>::Reset();
            routes_ = std::move(routes);
            operator_space_.Reset();
            cost_manager_.Reset();
        }

        Int EvaluateSolution() override {
            return problem_.GetTotalCost();
        }

        Int NumNeighbors() const override {
            return problem_.operator_space()->NumNeighbors();
        }

        bool IsTabu(Int neighbor_i) override {
            return false;
        }

        Int EvaluateNeighbor(Int neighbor_i) override {
            const Operator* op = problem_.operator_space()->GetNeighbor(neighbor_i);
            return problem_.EvaluateDelta(*op) + problem_.GetTotalCost();
        }

        std::optional<Int> Select() override {
            problem_.operator_space()->GenerateNeighbors();
            return rlop::TabuSearch<Int>::Select();
        }

        bool Step(const Int& neighbor_i) override {
            const Operator* op = problem_.operator_space()->GetNeighbor(neighbor_i);
            if (!problem_.Step(*op))
                return false;
            return true;
        }

        void RecordSolution() override {
            best_routes_ = (*problem_.routes());
        }

        const Routes& best_routes() const {
            return best_routes_;
        }

    protected:
        Routes routes_;
        Routes best_routes_;
        ArcCostManager cost_manager_;
        OperatorSpace operator_space_;
        Problem problem_;
    };
}