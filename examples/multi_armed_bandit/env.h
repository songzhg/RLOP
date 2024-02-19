#pragma once
#include "rlop/common/typedef.h"
#include "rlop/common/random.h"

namespace multi_armed_bandit {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class ActionSpace {
    public:
        ActionSpace(Int num_arms) : num_arms_(num_arms) {}

        virtual ~ActionSpace() = default;

        virtual Int Size() const {
            return num_arms_; 
        }

        virtual Int Get(Int i) const {
            return i;
        }

    protected:
        Int num_arms_ = 0;
    };

    class Env {
    public:
        Env(Int num_arms) : reward_dists_(num_arms), action_space_(num_arms) {}

        virtual ~Env() = default;

        virtual void Reset() {
            total_reward_ = 0;
            reward_dists_.clear();
            double best = std::numeric_limits<double>::lowest();
            for (Int i=0; i<action_space_.Size(); ++i) {
                double mean = rand_.Normal(0.0, 1.0);
                reward_dists_.push_back(std::pair{ mean, 1.0 });
                if (mean > best) {
                    best = mean;
                    best_arm_ = i;
                }
            }
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

        const ActionSpace& action_space() const {
            return action_space_;
        }

    protected:
        double total_reward_ = 0;
        Int num_arms_ = 0;
        Int best_arm_ = kIntNull;
        std::vector<std::pair<double, double>> reward_dists_;
        ActionSpace action_space_;
        rlop::Random rand_;
    };
}