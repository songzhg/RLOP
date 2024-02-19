#pragma once
#include "alpha_beta_search.h"

namespace rlop {
    template<typename TKey>
    class AlphaBetaSearchTrans : public AlphaBetaSearch {
    public:
        AlphaBetaSearchTrans(double max_score) : AlphaBetaSearch(max_score) {}

        virtual ~AlphaBetaSearchTrans() = default;

        virtual TKey PositionEncode() = 0;

        virtual std::optional<std::pair<double, ValueType>> Transpose(const TKey& key, Int depth) = 0;

        virtual void UpdateTable(const TKey& key, Int depth, double value, ValueType type) = 0;

        virtual double AlphaBeta(Int depth, double alpha, double beta) override {
            double origin_alpha = alpha;
            TKey key = PositionEncode();
            auto trans = Transpose(key, depth);
            if (trans) {
                auto [value, type] = *trans;
                if (type == ValueType::kExact) 
                    return value;
                else if (type == ValueType::kLowerBound)
                    alpha = std::max(alpha, value);
                else if (type == ValueType::kUpperBound)
                    beta = std::min(beta, value);
                if (alpha >= beta) 
                    return value;
            }
            if (depth == 0 || IsTerminal())
                return Evaluate();
            double value = -max_score_;
            std::vector<Int> moves = GenerateMoves();
            for (Int mv : moves) {
                if (!MakeMove(mv)) 
                    continue;
                value = std::max(value, -AlphaBeta(depth - 1, -beta, -alpha));
                UndoMove(mv);
                alpha = std::max(alpha, value);
                if (alpha >= beta) 
                    break;
            }
            ValueType type;
            if (value <= origin_alpha)
                type = ValueType::kUpperBound;
            if (value >= beta)
                type = ValueType::kLowerBound;
            else
                type = ValueType::kExact;
            UpdateTable(key, depth, value, type);
            return value;
        }
    };
}