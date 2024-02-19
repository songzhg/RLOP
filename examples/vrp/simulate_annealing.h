#pragma once
#include "problems/vrp/problem.h"
#include "rlop/local_search/simulated_annealing.h"
 
namespace vrp {
	class SimulatedAnnealing : public rlop::SimulatedAnnealing<Int, Int> {
	public:
		SimulatedAnnealing(
			Problem* problem,
			double initial_temp = 100,
			double final_temp = 0.01,
            double cooling_rate = 0.03
		) : 
			problem_(problem), 
			rlop::SimulatedAnnealing<Int, Int>(initial_temp, final_temp, cooling_rate)
		{}

		virtual ~SimulatedAnnealing() = default;

		virtual std::optional<Int> SelectRandom() override {
			if (problem_->operator_space()->NumNeighbors() == 0)
				return std::nullopt;
			return rand_.Uniform(Int(0), problem_->operator_space()->NumNeighbors() - 1);	
		}

		virtual std::optional<Int> SelectLocal() override {
			Int best = kIntNull;
            double best_score = std::numeric_limits<double>::max();
            for (Int i=0; i<problem_->operator_space()->NumNeighbors(); ++i) {
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

		virtual Int EvaluateSolution() override {
			return problem_->GetTotalCost();
		}

		virtual Int EvaluateNeighbor(const Int& op_i) override {
			auto op = problem_->operator_space()->GetNeighbor(op_i);
			return problem_->EvaluateDelta(*op) + problem_->GetTotalCost();
		}

		virtual std::optional<Int> Select() override {
			problem_->operator_space()->GenerateNeighbors();
			return rlop::SimulatedAnnealing<Int, Int>::Select();
		}

		virtual bool Step(const Int& op_i) override {
			auto op = problem_->operator_space()->GetNeighbor(op_i);
			if (!problem_->Step(*op))
				return false;
			return true;
		}

		virtual void RecordSolution() override {
			best_routes_ = *(problem_->routes());
		}

		const Routes& best_routes() const {
			return best_routes_;
		}

	protected:
		Problem* problem_ = nullptr;
		Routes best_routes_;
	};
}