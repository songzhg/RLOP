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

    // Continuous actions are usually considered to be independent, so we can sum components of the `log_prob` or the entropy.
    //
    // Parameters:
    //   tensor: A tensor with shape (n_batch, n_actions) or (n_batch,)
    // 
    // Returns:
    //   A tensor with shape (n_batch,) for an input tensor with shape (n_batch, n_actions),
    //   or a scalar for an input tensor with shape (n_batch,)
    inline torch::Tensor SumIndependentDims(const torch::Tensor& tensor) {
        if (tensor.sizes().size() > 1)
            return tensor.sum(1);
        else
            return tensor.sum();
    }

    inline torch::Tensor LogitsToProbs(const torch::Tensor& logits, bool is_binary = false) {
        if (is_binary)
            return torch::sigmoid(logits);
        else
            return torch::softmax(logits, -1);
    }

    class DiagGaussian : public RLDistribution {
    public:
        DiagGaussian(const torch::Tensor& mean, const torch::Tensor& std) : mean_(mean), std_(std) {}

        virtual torch::Tensor Sample() const override {
            torch::Tensor eps = torch::empty(mean_.sizes()).normal_().to(mean_.device());
            return eps * std_ + mean_;
        }

        virtual torch::Tensor Mode() const override {
            return mean_;
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

        virtual torch::Tensor Mode() const override {
            return torch::tanh(DiagGaussian::Mode());
        }

        virtual torch::Tensor LogProb(const torch::Tensor& x) const override {
            return torch::Tensor();
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

    class Categorical : public RLDistribution {
    public:
        Categorical(const torch::Tensor& logits) : logits_(logits - logits.logsumexp(-1, true)), probs_(LogitsToProbs(logits_)) {}

        virtual torch::Tensor Sample() const override {
            return torch::multinomial(probs_, 1).flatten();
        }

        virtual torch::Tensor Mode() const override {
            return std::get<1>(torch::max(probs_, -1));
        }

        virtual torch::Tensor LogProb(const torch::Tensor& x) const override {
            torch::Tensor value = x.to(torch::kInt64).unsqueeze(-1);
            auto broadcasted = torch::broadcast_tensors({value, logits_});
            value = broadcasted[0].index({"...", torch::indexing::Slice(0, 1)});
            return broadcasted[1].gather(-1, value).squeeze(-1);
        }

        torch::Tensor Entropy() const override {
            double min_real = 0;
            if (logits_.scalar_type() == c10::ScalarType::Float)
                min_real = std::numeric_limits<float>::lowest();
            else if (logits_.scalar_type() == c10::ScalarType::Double)
                min_real = std::numeric_limits<double>::lowest();
            torch::Tensor logits = torch::clamp(logits_, min_real);
            torch::Tensor p_log_p = logits * probs_;
            return -p_log_p.sum(-1);
        }

    protected:
        torch::Tensor logits_;
        torch::Tensor probs_;
    };
}