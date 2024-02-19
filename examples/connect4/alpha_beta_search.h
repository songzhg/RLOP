#pragma once
#include "problems/connect4/problem.h"
#include "rlop/minmax/alpha_beta_search_trans.h"
#include "rlop/minmax/transpositions.h"

namespace connect4 {
    class AlphaBetaSearch : public rlop::AlphaBetaSearchTrans<Board::bitboard> {
    public:
        static constexpr Int kTransSize = 8306069;
        static constexpr Int kSymmetryThres = 10;
        static constexpr double kWinScore = 1;

        AlphaBetaSearch() : rlop::AlphaBetaSearchTrans<Board::bitboard>(kWinScore), transposition_(kTransSize) {}

        void Reset() override {
            problem_.Reset();
            transposition_.Reset();
            ResetPriorScores();
        }

        void Reset(const Board& board) {
            problem_.Reset(board);
            transposition_.Reset();
            ResetPriorScores();
        } 

        void Reset(Board&& board) {
            problem_.Reset(std::move(board));
            transposition_.Reset();
            ResetPriorScores();
        }

        void ResetPriorScores() {
            prior_scores_ = {
                { 4, 6, 8, 10, 8, 6, 4 },
                { 5, 8, 11, 13, 11, 8, 5 },
                { 7, 10, 13, 16, 13, 10, 7 },
                { 5, 8, 11, 13, 11, 8, 5 },
                { 4, 6, 8, 10, 8, 6, 4 },
                { 3, 4, 5, 7, 5, 4, 3 }
            };
        }

        double Evaluate() override {
            if (problem_.board().Win())
                return -1;
            return 0;
        }

        bool IsTerminal() override {
            if (problem_.board().IsFull())
                return true;
            else if (problem_.board().Win()) 
                return true;

            return false;
        }

        Board::bitboard PositionEncode() override {
            Board::bitboard code = problem_.board().PositionEncode();
            if (problem_.board().num_moves() < kSymmetryThres) {
                Board::bitboard rev = 0; 
                for (auto tmp = code; tmp != 0; tmp >>= Board::kH1_) {
                    rev = (rev << Board::kH1_) | (tmp & Board::kCol1_);
                }
                if (rev < code)
                    code = rev;
            }
            return code;
        }
        
        Int GetPriorScore(Int move) const {
            return prior_scores_[problem_.board().heights()[move]][move];
        }

        std::vector<Int> GenerateMoves() override {
            std::vector<Int> moves;
            moves.reserve(problem_.NumMoves());
            for (Int i=0; i<problem_.NumMoves(); ++i) {
                Int move = problem_.GetMove(i);
                if (problem_.Step(move)) {
                    if (problem_.board().Win()) {
                        problem_.Undo(move);
                        moves = { move };
                        break;
                    }
                    problem_.Undo(move);
                    moves.push_back(move);
                }
            }
            std::sort(moves.begin(), moves.end(), [this](Int mv1, Int mv2){ return this->GetPriorScore(mv1) > this->GetPriorScore(mv2);});
            return moves;
        }

        bool MakeMove(Int move) override {
            return problem_.Step(move);
        }

        void UndoMove(Int move) override {
            problem_.Undo(move);
        }

        std::optional<std::pair<double, ValueType>> Transpose(const Board::bitboard& key, Int depth) override {
            const auto& item = transposition_.Get(key);
            if (item.lock == key && item.depth >= depth) 
                return std::pair<double, ValueType>{ item.value, item.type };
            return std::nullopt;
        }

        void UpdateTable(const Board::bitboard& key, Int depth, double value, ValueType type) override {
            const auto& item = transposition_.Get(key);
            if (item.type == ValueType::kNone || depth > item.depth)
                transposition_.Save(key, { key, depth, value, type });
        }

        auto NewSearch(const Board& board, Int depth = kIntFull) {
            Reset(board);
            auto [mv, value] = Search(depth);
            if (mv == kIntNull) {
                for (Int i=0; i<problem_.NumMoves(); ++i) {
                    Int mv = problem_.GetMove(i);
                    if (problem_.board().IsPlayable(mv))
                        return mv;
                }
            }
            return mv;
        }

    protected:
        Problem problem_;
        rlop::CircularTransposition<Board::bitboard> transposition_;
        std::vector<std::vector<Int>> prior_scores_;
    };
}