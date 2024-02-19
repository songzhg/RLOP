#pragma once
#include "rlop/common/torch_utils.h"

namespace rlop {
    class RLDistribution {
    public:
        virtual torch::Tensor Sample(const c10::ArrayRef<Int>& size) const = 0;

        virtual torch::Tensor LogProb(const torch::Tensor& value) const = 0;

        virtual torch::Tensor Entropy() const = 0;
    };

    class DiagGaussian : public RLDistribution {
    public:
        DiagGaussian(const torch::Tensor& mean, const torch::Tensor& log_std) : mean_(mean), log_std_(log_std) {}

        virtual torch::Tensor Sample(const c10::ArrayRef<Int>& size) const override {
            torch::Tensor standard = torch::randn(size).to(mean_.device());
            return standard * log_std_.exp() + mean_;
        }

        virtual torch::Tensor LogProb(const torch::Tensor& value) const override {
            auto log_scale = log_std_ + std::log(std::sqrt(2 * M_PI));
            return -0.5 * torch::pow((value - mean_) / log_std_.exp(), 2) - log_scale;
        }

        virtual torch::Tensor Entropy() const override {
            return log_std_ + 0.5 * std::log(2 * M_PI * M_E);
        }

    protected:
        torch::Tensor mean_;
        torch::Tensor log_std_;
    };

    class SquashedDiagGaussian : public DiagGaussian {
    public:
        SquashedDiagGaussian(const torch::Tensor& mean, const torch::Tensor& log_std, double eps = 1e-6) : DiagGaussian(mean, log_std), eps_(eps) {}
        
        virtual torch::Tensor Sample(const c10::ArrayRef<Int>& size) const override {
            return torch::tanh(DiagGaussian::Sample(size));
        }

        virtual torch::Tensor LogProb(const torch::Tensor& value) const override {
            torch::Tensor gaussian_value = torch_utils::Atanh(value);
            return DiagGaussian::LogProb(gaussian_value) - torch::sum(torch::log(1 - value.pow(2) + eps_), 1, true);
        }

        virtual torch::Tensor LogProb(const torch::Tensor& value, const torch::Tensor& gaussian_value) const {
            return DiagGaussian::LogProb(gaussian_value) - torch::sum(torch::log(1 - value.pow(2) + eps_), 1, true);
        }

        virtual torch::Tensor Entropy() const override {
            return torch::Tensor();
        }

    protected:
        double eps_;
    };

}