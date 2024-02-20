#pragma once
#include "problems/vrp/problem.h"
#include "rlop/local_search/simulated_annealing.h"
 
namespace vrp {
    class SimulatedAnnealing : public rlop::SimulatedAnnealing<Int, Int> {
    public:
        SimulatedAnnealing(
			const std::function<Int(Int,Int)>& get_cost,
            double initial_temp = 100,
            double final_temp = 0.01,
            double cooling_rate = 0.03
        ) : 
            rlop::SimulatedAnnealing<Int, Int>(initial_temp, final_temp, cooling_rate),
			operator_space_(routes_),
			cost_manager_(routes_, get_cost),
            problem_(&routes_, &operator_space_, { &cost_manager_ })
        {}

		void Reset() override {
			rlop::SimulatedAnnealing<Int, Int>::Reset();
			routes_.Reset();
			operator_space_.Reset();
			cost_manager_.Reset();
		}

		void Reset(const Routes& routes) {
			rlop::SimulatedAnnealing<Int, Int>::Reset();
			routes_ = routes;
			operator_space_.Reset();
			cost_manager_.Reset();
		}

		void Reset(Routes&& routes) {
			rlop::SimulatedAnnealing<Int, Int>::Reset();
			routes_ = std::move(routes);
			operator_space_.Reset();
			cost_manager_.Reset();
		}

        std::optional<Int> SelectRandom() override {
            if (problem_.operator_space()->NumNeighbors() == 0)
                return std::nullopt;
            return rand_.Uniform(Int(0), problem_.operator_space()->NumNeighbors() - 1);    
        }

        std::optional<Int> SelectLocal() override {
            Int best = kIntNull;
            double best_score = std::numeric_limits<double>::max();
            for (Int i=0; i<problem_.operator_space()->NumNeighbors(); ++i) {
                double score = EvaluateNeighbor(i);
                if (score < best_score) {
                    best = i;
                    best_score = score;
                }
            }
            if (best == kIntNull)
                return std::nullopt;
            return { best }; 
        }

        Int EvaluateSolution() override {
            return problem_.GetTotalCost();
        }

        Int EvaluateNeighbor(const Int& op_i) override {
            auto op = problem_.operator_space()->GetNeighbor(op_i);
            return problem_.EvaluateDelta(*op) + problem_.GetTotalCost();
        }

        std::optional<Int> Select() override {
            problem_.operator_space()->GenerateNeighbors();
            return rlop::SimulatedAnnealing<Int, Int>::Select();
        }

        bool Step(const Int& op_i) override {
            auto op = problem_.operator_space()->GetNeighbor(op_i);
            if (!problem_.Step(*op))
                return false;
            return true;
        }

        void RecordSolution() override {
            best_routes_ = routes_;
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