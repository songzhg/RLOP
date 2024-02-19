#pragma once
#include "random.h"
#include "utils.h"

namespace rlop {
    template<typename TIterator>
    std::optional<Int> SelectBest(const TIterator& score_begin, const TIterator& score_end) {
        using TScore = typename std::iterator_traits<TIterator>::value_type;
        Int best = kIntNull;
        TScore best_score = std::numeric_limits<TScore>::lowest();
        for (auto it=score_begin; it!=score_end; ++it) {
            TScore score = *it;
            if (score > best_score) {
                best = it - score_begin;
                best_score = score;
            }
        }
        if (best == kIntNull)
            return std::nullopt;
        return { best };
    }

    template<typename TIterator>
    std::optional<Int> SelectRoundRobin(const TIterator& mask_begin, const TIterator& mask_end, Int current) {
        using TMask = typename std::iterator_traits<TIterator>::value_type;
        static_assert(std::is_same_v<bool, TMask>, "SelectRoundRobin: requires a bool mask type.");
        if (current >= mask_end - mask_begin)
            return std::nullopt;
        auto current_it = mask_begin + current;
        auto it = current_it;
        while (true) {
            ++it;
            if (it == mask_end)
                it = mask_begin;
            if (*it)
                break;
            else if (it == current_it)
                return std::nullopt;
        }
        return { it - mask_begin };
    }

    template<typename TIterator>
    std::optional<Int> SelectUniform(const TIterator& mask_begin, const TIterator& mask_end, Random* rand) {
        using TMask = typename std::iterator_traits<TIterator>::value_type;
        static_assert(std::is_same_v<bool, TMask>, "SelectRoundRobin: requires a bool mask type.");
        std::vector<TIterator> indices;
        indices.reserve(mask_end - mask_begin);
        for (auto it=mask_begin; it != mask_end; ++it) {
            if (*it)
                indices.push_back(it);
        }
        if (indices.empty())
            return std::nullopt;
        return { indices[rand->Uniform(size_t(0), indices.size() - 1)] - mask_begin };
    }
}