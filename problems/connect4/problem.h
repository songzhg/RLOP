#pragma once
#include "board.h"
#include "rlop/common/typedef.h"

namespace connect4 {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class Problem {
    public:
        Problem() = default;

        virtual ~Problem() = default; 

        virtual void Reset() {
            board_.Reset();
        }

        virtual void Reset(const Board& board) {
            board_ = board;
        }

        virtual void Reset(Board&& board) {
            board_ = std::move(board);
        }

        virtual Int NumMoves() const {
            return Board::kWidth_;
        }

        virtual Int GetMove(Int i) const {
            return i;
        }

        virtual bool Step(Int move) {
            return board_.MakeMove(move);
        }

        virtual void Undo(Int move) {
            board_.UndoMove(move);
        }

        const Board& board() const {
            return board_;
        }

    protected:
        Board board_;
    };

    class VectorProblem {
    public:
        VectorProblem(Int num_problems) : boards_(num_problems) {}
        
        virtual ~VectorProblem() = default; 

        virtual void Reset(Int env_i) {
            boards_[env_i].Reset();
        }

        virtual void Reset(Int env_i, const Board& board) {
            boards_[env_i] = board;
        }

        virtual void Reset(Int env_i, Board&& board) {
            boards_[env_i] = std::move(board);
        }

        virtual Int NumMoves() const {
            return Board::kWidth_;
        }

        virtual Int GetMove(Int i) const {
            return i;
        }

        virtual bool Step(Int env_i, Int move) {
            return boards_[env_i].MakeMove(move);
        }

        virtual void Undo(Int env_i, Int move) {
            boards_[env_i].UndoMove(move);
        }

        const std::vector<Board>& boards() const {
            return boards_;
        }

        Int num_problems() const {
            return boards_.size();
        }

    protected:
        std::vector<Board> boards_;
    };
}