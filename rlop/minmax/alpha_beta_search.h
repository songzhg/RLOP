#pragma once
#include "rlop/common/base_algorithm.h"

namespace rlop {
    // Implements the Alpha-Beta pruning algorithm, an optimization of the Minimax algorithm
    // for searching game trees. It is designed to reduce the number of nodes evaluated in the
    // search tree by pruning branches that are not likely to lead to better solutions than those
    // already found.
    class AlphaBetaSearch : public BaseAlgorithm {
    public:
        // Defines the type of value returned by the AlphaBeta search.
        enum class ValueType {
            kExact = 0,
            kLowerBound,
            kUpperBound,
            kNone
        };

        // Constructor initializes the algorithm with the maximum possible score in the game.
        //
        // Parameters:
        //   max_score: The maximum score a player can achieve in the game. This is used to initialize
        //              the alpha and beta values for the root of the search tree.
        AlphaBetaSearch(double max_score) : max_score_(max_score) {}

        virtual ~AlphaBetaSearch() = default; 

        // Pure virtual function to evaluate the current game state and return a score.
        virtual double Evaluate() = 0;

        // Pure virtual function to check if the current game state is terminal (i.e., the game is over). 
        virtual bool IsTerminal() = 0;

        // Pure virtual function to generate all possible moves from the current game state.
        virtual std::vector<Int> GenerateMoves() = 0;

        // Pure virtual function tp make a move and updates the game state accordingly.
        //
        // Parameters:
        //   move: The move to make.
        //
        // Returns:
        //   bool: True if the move was made successfully, false otherwise.
        virtual bool MakeMove(Int move) = 0;
        
        // Pure virtual function to undoe a move, reverting the game state to its previous state.
        //
        // Parameters:
        //   move: The move to undo.
        virtual void UndoMove(Int move) = 0;

        // Implements the Alpha-Beta pruning algorithm for a given depth, alpha, and beta values.
        //
        // Parameters:
        //   depth: The maximum depth of the search tree to explore.
        //   alpha: The current alpha value, representing the minimum score that the maximizing player is assured of.
        //   beta: The current beta value, representing the maximum score that the minimizing player is assured of.
        //
        // Returns:
        //   double: The value of the node.
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

        // Initiates the Alpha-Beta search algorithm to find the best move and its value from the current game state.
        //
        // Parameters:
        //   depth: The maximum depth of the search tree to explore.
        //   alpha: The initial alpha value (optional).
        //   beta: The initial beta value (optional).
        //
        // Returns:
        //   std::pair<Int, double>: The best move and its value. If no move is found, returns {kIntNull, best_value}.
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