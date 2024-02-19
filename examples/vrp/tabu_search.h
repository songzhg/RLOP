#pragma once
#include "problems/vrp/problem.h"
#include "rlop/local_search/tabu_search.h"
 
namespace vrp {
	class TabuSearch : public rlop::TabuSearch<Int> {
	public:
		TabuSearch(
			Problem* problem,
			Int max_num_unimproved_iters = 50, 
			Int tenure = 10 
		) : 
			rlop::TabuSearch<Int>(max_num_unimproved_iters),
			problem_(problem), 
			tenure_(tenure)
		{}

		virtual ~TabuSearch() = default;

		virtual void Reset() override {
			rlop::TabuSearch<Int>::Reset();
			tabu_table_.Reset();
		}

		virtual Int EvaluateSolution() override {
			return problem_->GetTotalCost();
		}

		virtual Int NumNeighbors() const override {
			return problem_->operator_space()->NumNeighbors();
		}

		virtual bool IsTabu(Int neighbor_i) override {
			const Operator* op = problem_->operator_space()->GetNeighbor(neighbor_i);
			return tabu_table_.IsTabu(problem_->EncodeOperator(*op));
		}

		virtual Int EvaluateNeighbor(Int neighbor_i) override {
			const Operator* op = problem_->operator_space()->GetNeighbor(neighbor_i);
			return problem_->EvaluateDelta(*op) + problem_->GetTotalCost();;
		}

		virtual std::optional<Int> Select() override {
			problem_->operator_space()->GenerateNeighbors();
			return rlop::TabuSearch<Int>::Select();
		}

		virtual bool Step(const Int& neighbor_i) override {
			const Operator* op = problem_->operator_space()->GetNeighbor(neighbor_i);
			if (!problem_->Step(*op))
				return false;
			tabu_table_.Tabu(problem_->EncodeOperator(*op), tenure_);
			return true;
		}

		virtual void RecordSolution() override {
			best_routes_ = *(problem_->routes());
		}

		virtual void Update() override {
			rlop::TabuSearch<Int>::Update();
			tabu_table_.Update();
		}

		Int tenure() const {
			return tenure_;
		}

		const Routes& best_routes() const {
			return best_routes_;
		}

		void set_tenure(Int num) {
			tenure_ = num;	
		}

	protected:
		Problem* problem_ = nullptr;
		Routes best_routes_;
		Int tenure_;
		rlop::HashTabuTable<Int> tabu_table_;
	};
}