#pragma once
#include <random>
#include "typedef.h"

namespace rlop {
    class Random {
    public:
        Random() : gen_() {}

        Random(uint64_t seed) : gen_(seed) {}

        void Seed(uint64_t seed) {
            gen_.seed(seed);
        }
        
        template <typename T>
        T Uniform(T min_value, T max_value) {
            static_assert(std::is_arithmetic_v<T>, "Random: uniform requires an arithmetic type.");
            if constexpr (std::is_integral_v<T>) {
                static std::uniform_int_distribution<T> dist;
                return dist(gen_, typename std::uniform_int_distribution<T>::param_type(min_value, max_value));
            } else {
                static std::uniform_real_distribution<T> dist;
                return dist(gen_, typename std::uniform_real_distribution<T>::param_type(min_value, max_value));
            }
        }

        template <typename T>
        T Normal(T mean, T std) {
            static_assert(std::is_arithmetic_v<T>, "Random: normal requires an arithmetic type.");
            static std::normal_distribution<T> dist;
            return dist(gen_, typename std::normal_distribution<T>::param_type(mean, std));
        }

        template <typename T>
        T Poisson(T mean) const {
            static_assert(std::is_arithmetic_v<T>, "Random: poisson requires an arithmetic type.");
            static std::poisson_distribution<T> dist;
            return dist(gen_, typename std::poisson_distribution<T>::param_type(mean));
        }

        template <typename T, typename TIterator>
        T Discrete(const TIterator& begin, const TIterator& end) {
            static_assert(std::is_integral_v<T>, "Random: discrete requires an integral type");
            static_assert(std::is_same_v<TIterator, typename std::vector<typename TIterator::value_type>::iterator> || 
                std::is_same_v<TIterator, typename std::vector<typename TIterator::value_type>::const_iterator>, "Random: discrete requires a vector iterator type.");
            static std::discrete_distribution<T> dist;
            return dist(gen_, typename std::discrete_distribution<T>::param_type(begin, end));
        }

        template<typename TIterator>
        void PartialShuffle(const TIterator& begin, const TIterator& end, Int n) {
            static_assert(std::is_same_v<TIterator, typename std::vector<typename TIterator::value_type>::iterator>, "Random: partial shuffle requires a vector iterator type.");
            Int size = static_cast<Int>(std::distance(begin, end));
            for (Int i=0; i<n; ++i) {
                Int selected = Uniform(i, size - 1);
                std::iter_swap(begin + i, begin + selected);
            }
        }
        
        template<typename TIterator>
        void Shuffle(const TIterator& begin, const TIterator& end) {
            static_assert(std::is_same_v<TIterator, typename std::vector<typename TIterator::value_type>::iterator>, "Random: shuffle requires a vector iterator type.");
            std::shuffle(begin, end, gen_);   
        }

    protected:
        std::mt19937_64 gen_;
    };
}