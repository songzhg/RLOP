#pragma once
#include "typedef.h"

namespace rlop {
    inline std::vector<std::string> SplitStr(const std::string& input, const std::string& delimiter) {
        std::vector<std::string> result;
        size_t start_pos = 0;
        size_t found_pos = input.find(delimiter);
        while (found_pos != std::string::npos) {
            result.push_back(input.substr(start_pos, found_pos - start_pos));
            start_pos = found_pos + delimiter.length();
            found_pos = input.find(delimiter, start_pos);
        }
        result.push_back(input.substr(start_pos));
        return result;
    }

    inline std::string StripStr(const std::string& input) {
        size_t start = input.find_first_not_of(" \t\n");
        size_t end = input.find_last_not_of(" \t\n");
        if (start == std::string::npos) 
            return "";
        return input.substr(start, end - start + 1);
    }

    inline double UCB1(double q_value, Int num_visits, Int total_num_visits, double c = std::sqrt(2)) {
        if (num_visits == 0)
            return std::numeric_limits<double>::max();
        return q_value + c * std::sqrt(std::log(total_num_visits) / num_visits);
    }

    template<typename TIterator>
    std::vector<double> Softmax(const TIterator& begin, const TIterator& end, double temp = 1.0) {
        using TScore = typename std::iterator_traits<TIterator>::value_type;
        static_assert(std::is_arithmetic_v<TScore>, "Softmax: requires arithmetic iterator type.");
        std::vector<double> output(end - begin);
        double max_input = *std::max_element(begin, end);
        std::transform(begin, end, output.begin(), [max_input](double x) { return std::exp(x - max_input); });
        double sum = std::accumulate(output.begin(), output.end(), 0.0);
        std::transform(output.begin(), output.end(), output.begin(), [sum](double x) { return x / sum; });
        return output;
    }

    template<typename TIterator>
    double ComputeMean(const TIterator& begin, const TIterator& end) {
        using TValue = typename std::iterator_traits<TIterator>::value_type;
        static_assert(std::is_arithmetic_v<TValue>, "ComputeMean: requires arithmetic iterator type.");
        if (begin == end)
            return 0;
        double sum = std::accumulate(begin, end, 0.0);
        return sum / (end - begin);
    }

    template<typename TIterator>
    double ComputeVariance(const TIterator& begin, const TIterator& end, double mean) {
        using TValue = typename std::iterator_traits<TIterator>::value_type;
        static_assert(std::is_arithmetic_v<TValue>, "ComputeVariance: requires arithmetic iterator type.");
        Int size = end - begin;
        if (size < 2) 
            return 0.0;
        double sum = 0.0;
        for (auto it = begin; it != end; ++it) {
            sum += (*it - mean) * (*it - mean);
        }
        return sum / (size - 1);
    }

    inline std::function<double(double x)> MakeLinearFn(double start, double end, double end_fraction) {
        return [start, end, end_fraction](double current_fraction) {
            if (current_fraction > end_fraction)
                return end;
            else
                return start + current_fraction * (end - start) / end_fraction;
        };
    }
}