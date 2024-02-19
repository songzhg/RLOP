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

        CircularTransposition(size_t size) : table_(size) {}

        virtual ~CircularTransposition() = default;

        virtual void Reset() {
            table_ = std::vector<Item>(table_.size());
        }

        virtual void Save(TKey key, const Item& item) {
            table_[key % table_.size()] = item;
        }

        virtual void Save(TKey key, Item&& item) {
            table_[key % table_.size()] = std::move(item);
        }

        virtual const Item& Get(TKey key) {
            return table_[key % table_.size()];
        }

    protected:
        std::vector<Item> table_;
    };
}