#pragma once
#include "rlop/common/torch_utils.h"

namespace rlop {
    class RLDistribution {
    public:
        virtual torch::Tensor Sample() const = 0;

        virtual torch::Tensor Mode() const = 0;

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

        virtual torch::Tensor Mode() const override {
            return mean_;
        }

        virtual torch::Tensor LogProb(const torch::Tensor& x) const override {
            torch::Tensor log_prob = -(x - mean_).square() / (2.0 * std_.square()) - std_.log() - std::log(std::sqrt(2.0 * M_PI));
            return torch_utils::SumIndependentDims(log_prob);
        }

        virtual torch::Tensor Entropy() const override {
            auto entropy = 0.5 + 0.5 * std::log(2 * M_PI) + std_.log();
            return torch_utils::SumIndependentDims(entropy);
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

        virtual torch::Tensor Mode() const override {
            return torch::tanh(DiagGaussian::Mode());
        }

        virtual torch::Tensor LogProb(const torch::Tensor& x) const override {
            return torch::Tensor();
        }

        virtual torch::Tensor LogProb(const torch::Tensor& x, const torch::Tensor& gaussian_x) const {
            return DiagGaussian::LogProb(gaussian_x) - torch_utils::SumIndependentDims(torch::log(1.0 - x.square() + eps_));
        }

        virtual torch::Tensor Entropy() const override {
            return torch::Tensor();
        }

    protected:
        double eps_;
    };

    class Categorical : public RLDistribution {
    public:
        Categorical(const torch::Tensor& logits, double eps = 1e-8) : logits_(logits), prob_(torch_utils::LogitsToProbs(logits)), eps_(eps) {}

        virtual torch::Tensor Sample() const override {
            return torch::multinomial(prob_, 1).flatten();
        }

        virtual torch::Tensor Mode() const override {
            return std::get<1>(torch::max(prob_, -1));
        }

        virtual torch::Tensor LogProb() const {
            if (!log_prob_.defined())
                log_prob_ = torch::log(prob_ + eps_);
            return log_prob_;
        }

        virtual torch::Tensor LogProb(const torch::Tensor& x) const override {
            return torch::gather(LogProb(), 1, x.reshape({-1, 1})).flatten();
        }

        virtual torch::Tensor Entropy() const override {
            return -torch::sum(prob_ * LogProb(), -1);;
        } 

    protected:
        torch::Tensor logits_;
        torch::Tensor prob_;
        mutable torch::Tensor log_prob_;
        double eps_;
    };
}