#pragma once
#include "problem.h"
#include "rlop/common/selectors.h"
#include "rlop/common/base_algorithm.h"
 
namespace vrp {
    class InsertionSolver : public rlop::BaseAlgorithm {
	public:
		InsertionSolver(Problem* problem) : problem_(problem) {}

		~InsertionSolver() = default;

		virtual void Reset() override {}

		virtual Int Evaluate(Int i) const {
			auto op = problem_->operator_space()->GetInsertion(i);
			return problem_->EvaluateDelta(*op);
		}

		virtual std::optional<Int> Select() {
            Int best = kIntNull;
            double best_score = std::numeric_limits<double>::max();
            for (Int i=0; i<problem_->operator_space()->NumInsertions(); ++i) {
                double score = Evaluate(i);
                if (score < best_score) {
                    best = i;
                    best_score = score;
                }
            }
            if (best == kIntNull)
                return std::nullopt;
            return { best };
        }

		virtual void Solve() {
			while (true) {
				auto i = Select();
				if (!i)
					return;
				auto op = problem_->operator_space()->GetInsertion(*i);
				if (!problem_->Step(*op))
					return;
			}
		}

	protected:
		Problem* problem_ = nullptr;
	};
}