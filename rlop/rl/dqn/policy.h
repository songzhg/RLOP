#pragma once
#include "rlop/rl/policy.h"

namespace rlop {
    class QNet : public RLPolicy {
    public:
        QNet() = default;

        virtual ~QNet() = default;

        virtual torch::Tensor PredictAction(const torch::Tensor& observation, bool deterministic = false) override {
            return std::get<1>(torch::max(Forward(observation), -1));
        }

        virtual torch::Tensor Forward(const torch::Tensor& observation) = 0;
    };
}