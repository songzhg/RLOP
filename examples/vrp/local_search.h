#pragma once
#include "problems/vrp/problem.h"
#include "rlop/local_search/tabu_search.h"
 
namespace vrp {
	class LocalSearch : public rlop::TabuSearch<Int> {
	public:
		LocalSearch(
			Problem* problem,
			Int max_num_unimproved_iters = 50 
		) : 
			rlop::TabuSearch<Int>(max_num_unimproved_iters),
			problem_(problem) 
		{}

		virtual ~LocalSearch() = default;


		virtual Int EvaluateSolution() override {
			return problem_->GetTotalCost();
		}

		virtual Int NumNeighbors() const override {
			return problem_->operator_space()->NumNeighbors();
		}

		virtual bool IsTabu(Int neighbor_i) override {
			return false;
		}

		virtual Int EvaluateNeighbor(Int neighbor_i) override {
			const Operator* op = problem_->operator_space()->GetNeighbor(neighbor_i);
			return problem_->EvaluateDelta(*op) + problem_->GetTotalCost();
		}

		virtual std::optional<Int> Select() override {
			problem_->operator_space()->GenerateNeighbors();
			return rlop::TabuSearch<Int>::Select();
		}

		virtual bool Step(const Int& neighbor_i) override {
			const Operator* op = problem_->operator_space()->GetNeighbor(neighbor_i);
			if (!problem_->Step(*op))
				return false;
			return true;
		}

		virtual void RecordSolution() override {
			best_routes_ = (*problem_->routes());
		}

		const Routes& best_routes() const {
			return best_routes_;
		}

	protected:
		Problem* problem_ = nullptr;
		Routes best_routes_;
	};
}