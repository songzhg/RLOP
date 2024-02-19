#pragma once
#include "rlop/common/base_algorithm.h"

namespace rlop {
    class AlphaBetaSearch : public BaseAlgorithm {
    public:
        enum class ValueType {
            kExact = 0,
            kLowerBound,
            kUpperBound,
            kNone
        };

        AlphaBetaSearch(double max_score) : max_score_(max_score) {}

        virtual ~AlphaBetaSearch() = default; 

        virtual double Evaluate() = 0;

        virtual bool IsTerminal() = 0;

        virtual std::vector<Int> GenerateMoves() = 0;

        virtual bool MakeMove(Int move) = 0;
        
        virtual void UndoMove(Int move) = 0;

        virtual double AlphaBeta(Int depth, double alpha, double beta) {
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
            return value;
        }

        virtual std::pair<Int, double> Search(Int depth, double alpha = std::numeric_limits<double>::lowest(), double beta = std::numeric_limits<double>::max()) {
            if (depth == 0 || IsTerminal())
                return { kIntNull, Evaluate() };
            alpha = std::max(-max_score_, alpha);
            beta = std::min(max_score_, beta); 
            Int best_mv = kIntNull;
            double best_value = -max_score_;
            std::vector<Int> moves = GenerateMoves();
            for (Int mv : moves) {
                if (!MakeMove(mv))
                    continue;
                double value = -AlphaBeta(depth - 1, -beta, -alpha);
                UndoMove(mv);
                if (value > best_value) {
                    best_value = value;
                    best_mv = mv;
                }
                alpha = std::max(alpha, best_value);
                if (alpha >= beta) 
                    break;
            } 
            return { best_mv, best_value }; 
        }

    protected:
        double max_score_; 
    };
}