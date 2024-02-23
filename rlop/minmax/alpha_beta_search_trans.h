#pragma once
#include "alpha_beta_search.h"

namespace rlop {
    // Extends the AlphaBetaSearch class to include transposition table support, optimizing the search process by storing
    // and reusing the results of previously evaluated positions. This template class allows for a customizable key type,
    // enabling the use of various methods for encoding game states into keys suitable for transposition table lookups.
    template<typename TKey>
    class AlphaBetaSearchTrans : public AlphaBetaSearch {
    public:
        // Constructor: Initializes the AlphaBeta search with a maximum score value.
        //
        // Parameters:
        //   max_score: The maximum possible score in the game, used to initialize alpha-beta bounds.
        AlphaBetaSearchTrans(double max_score) : AlphaBetaSearch(max_score) {}

        virtual ~AlphaBetaSearchTrans() = default;

        // Pure virtual function to encode the current position into a key of type TKey for transposition table entries
        virtual TKey PositionEncode() = 0;

        // Pure virtual function to attempt to retrieve a value and its type from the transposition table using the given
        // key and depth. This method is used to check if the current position (encoded as a key) at a specific depth has 
        // been previously evaluated and stored in the transposition table.
        //
        // Parameters:
        //   key: The encoded key representing the current game state.
        //   depth: The depth of the search at which the game state is evaluated.
        //
        // Returns:
        //   std::optional<std::pair<double, ValueType>>: An optional containing the value and its type if the position
        //                                                is found in the table; std::nullopt otherwise.
        virtual std::optional<std::pair<double, ValueType>> Transpose(const TKey& key, Int depth) = 0;

        // Pure virtual function to update the transposition table with a new value, its type, and the corresponding key
        // and depth. This method is called after evaluating a position to store the result for future reference, potentially
        // saving computation time by avoiding re-evaluation of the same position.
        //
        // Parameters:
        //   key: The key representing the game state being updated.
        //   depth: The depth at which the value was evaluated.
        //   value: The evaluated value of the position.
        //   type: The type of the value (exact, lower bound, or upper bound).
        virtual void UpdateTable(const TKey& key, Int depth, double value, ValueType type) = 0;

        // Overrides the AlphaBeta method to integrate transposition table lookups and updates, enhancing the search efficiency 
        // by reusing results of previously evaluated positions. This implementation checks the transposition table before 
        // proceeding with the standard AlphaBeta search logic. There is a discussion on how to combine negamax and transposition 
        // table correctly: https://en.wikipedia.org/wiki/Talk:Negamax.
        //
        // Parameters:
        //   depth: The depth of the search.
        //   alpha: The current lower bound of the best possible score.
        //   beta: The current upper bound of the best possible score.
        //
        // Returns:
        //   double: The evaluated value of the current position.
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