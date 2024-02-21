#pragma once
#include "alpha_beta_search_trans.h"

namespace rlop {
    template<typename TKey>
    class CircularTransposition {
    public:
        using Type = typename AlphaBetaSearch::ValueType;

        struct Item {
            TKey lock;
            Int depth;
            double value;
            Type type = Type::kNone;
        };

        CircularTransposition(size_t size) : vec_(size) {}

        virtual ~CircularTransposition() = default;

        virtual void Reset() {
            vec_ = std::vector<Item>(vec_.size());
        }

        virtual void Save(TKey key, const Item& item) {
            vec_[key % vec_.size()] = item;
        }

        virtual void Save(TKey key, Item&& item) {
            vec_[key % vec_.size()] = std::move(item);
        }

        virtual const Item& Get(TKey key) {
            return vec_[key % vec_.size()];
        }

        const std::vector<Int>& vec() const {
            return vec_;
        } 

    protected:
        std::vector<Item> vec_;
    };
}