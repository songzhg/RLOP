#pragma once
#include "rlop/common/torch_utils.h"

namespace rlop {
    inline torch::Tensor SumIndependentDims(const torch::Tensor& tensor) {
        if (tensor.sizes().size() > 1)
            return tensor.sum(1);
        else
            return tensor.sum();
    }

    class RLDistribution {
    public:
        virtual torch::Tensor Sample() const = 0;

        virtual torch::Tensor LogProb(const torch::Tensor& x) const = 0;

        virtual torch::Tensor Entropy() const = 0;
    };

    class DiagGaussian : public RLDistribution {
    public:
        DiagGaussian(const torch::Tensor& mean, const torch::Tensor& std) : mean_(mean), std_(std) {}

        virtual torch::Tensor Sample() const override {
            torch::Tensor eps = torch::empty_like(std_).normal_();
            return eps * std_ + mean_;
        }

        virtual torch::Tensor LogProb(const torch::Tensor& x) const override {
            torch::Tensor log_prob = -(x - mean_).square() / (2.0 * std_.square()) - std_.log() - std::log(std::sqrt(2.0 * M_PI));
            return SumIndependentDims(log_prob);
        }

        virtual torch::Tensor Entropy() const override {
            auto entropy = 0.5 + 0.5 * std::log(2 * M_PI) + std_.log();
            return SumIndependentDims(entropy);
        }

    protected:
        torch::Tensor mean_;
        torch::Tensor std_;
    };

    class SquashedDiagGaussian : public DiagGaussian {
    public:
        SquashedDiagGaussian(const torch::Tensor& mean, const torch::Tensor& std, double eps = 1e-6) : DiagGaussian(mean, std), eps_(eps) {}
        
        virtual torch::Tensor Sample() const override {
            return torch::tanh(DiagGaussian::Sample());
        }

        virtual torch::Tensor LogProb(const torch::Tensor& x, const torch::Tensor& gaussian_x) const {
            return DiagGaussian::LogProb(gaussian_x) - SumIndependentDims(torch::log(1.0 - x.square() + eps_));
        }

        virtual torch::Tensor Entropy() const override {
            return torch::Tensor();
        }

    protected:
        double eps_;
    };

}