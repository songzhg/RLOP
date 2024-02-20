#pragma once
#include "rlop/common/typedef.h"
#include "rlop/common/random.h"

namespace multi_armed_bandit {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class Problem {
    public:
        Problem(Int num_arms) : num_arms_(num_arms), reward_dists_(num_arms) {}

        virtual ~Problem() = default;

        virtual void Reset() {
            total_reward_ = 0;
            reward_dists_.clear();
            double best = std::numeric_limits<double>::lowest();
            for (Int i=0; i<num_arms_; ++i) {
                double mean = rand_.Normal(0.0, 1.0);
                reward_dists_.push_back(std::pair{ mean, 1.0 });
                if (mean > best) {
                    best = mean;
                    best_arm_ = i;
                }
            }
        }

        virtual Int NumActions() const {
            return num_arms_;
        }

        virtual Int GetAction(Int i) const {
            return i;
        }

        virtual void Reset(uint64_t seed) {
            Reset();
            rand_.Seed(seed);
        }

        virtual double Step(Int i) {
            double reward = rand_.Normal(reward_dists_[i].first, reward_dists_[i].second); 
            total_reward_ += reward;
            return reward;
        }

        double total_reward() const {
            return total_reward_; 
        } 

        Int num_arms() const {
            return reward_dists_.size();
        }

        Int best_arm() const {
            return best_arm_;
        }

        const std::vector<std::pair<double, double>>& reward_dists() const {
            return reward_dists_;
        }

    protected:
        double total_reward_ = 0;
        Int num_arms_ = 0;
        Int best_arm_ = kIntNull;
        std::vector<std::pair<double, double>> reward_dists_;
        rlop::Random rand_;
    };
}