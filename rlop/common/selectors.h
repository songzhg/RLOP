#pragma once
#include "random.h"
#include "utils.h"

namespace rlop {
    // Selects the index of the highest scoring element from a range of scores.
    //
    // Template Parameters:
    //   TIterator: Iterator type that points to the collection of scores.
    //
    // Parameters:
    //   begin: Iterator to the beginning of the score collection.
    //   end: Iterator to the end of the score collection.
    //
    // Returns:
    //   std::optional<Int>: The index of the element with the highest score. Returns std::nullopt if the
    //                       collection is empty or no valid score is found.
    template<typename TIterator>
    std::optional<Int> SelectBest(const TIterator& begin, const TIterator& end) {
        using TScore = typename std::iterator_traits<TIterator>::value_type;
        Int best = kIntNull;
        TScore best_score = std::numeric_limits<TScore>::lowest();
        for (auto it=begin; it!=end; ++it) {
            TScore score = *it;
            if (score > best_score) {
                best = it - begin;
                best_score = score;
            }
        }
        if (best == kIntNull)
            return std::nullopt;
        return { best };
    }

    // Selects the next true element in a boolean mask collection in a round-robin fashion.
    //
    // Template Parameters:
    //   TIterator: Iterator type pointing to the collection of boolean masks.
    //
    // Parameters:
    //   begin: Iterator to the beginning of the mask collection.
    //   end: Iterator to the end of the mask collection.
    //   current: The current index position from which to start the round-robin selection.
    //
    // Returns:
    //   std::optional<Int>: The index of the next true element after 'current' in the mask collection.
    //                       Returns std::nullopt if no such element exists or if 'current' is out of bounds.
    template<typename TIterator>
    std::optional<Int> SelectRoundRobin(const TIterator& begin, const TIterator& end, Int current) {
        using TMask = typename std::iterator_traits<TIterator>::value_type;
        static_assert(std::is_same_v<bool, TMask>, "SelectRoundRobin: requires a bool mask type.");
        if (current >= end - begin)
            return std::nullopt;
        auto current_it = begin + current;
        auto it = current_it;
        while (true) {
            ++it;
            if (it == end)
                it = begin;
            if (*it)
                break;
            else if (it == current_it)
                return std::nullopt;
        }
        return { it - begin };
    }

    // Selects a random index from a boolean mask collection where the mask is true.
    //
    // Template Parameters:
    //   TIterator: Iterator type pointing to the collection of boolean masks.
    //
    // Parameters:
    //   begin: Iterator to the beginning of the mask collection.
    //   end: Iterator to the end of the mask collection.
    //   rand: Pointer to a Random object for generating random numbers.
    //
    // Returns:
    //   std::optional<Int>: The index of a randomly selected true element in the mask collection.
    //                       Returns std::nullopt if no true elements are found.
    template<typename TIterator>
    std::optional<Int> SelectUniform(const TIterator& begin, const TIterator& end, Random* rand) {
        using TMask = typename std::iterator_traits<TIterator>::value_type;
        static_assert(std::is_same_v<bool, TMask>, "SelectRoundRobin: requires a bool mask type.");
        std::vector<TIterator> indices;
        indices.reserve(end - begin);
        for (auto it=begin; it != end; ++it) {
            if (*it)
                indices.push_back(it);
        }
        if (indices.empty())
            return std::nullopt;
        return { indices[rand->Uniform(size_t(0), indices.size() - 1)] - begin };
    }
}