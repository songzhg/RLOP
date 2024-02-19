#pragma once
#include "rlop/common/utils.h"

namespace rlop {
    template<typename TKey>
    class HashTabuTable {
    public:
        virtual ~HashTabuTable() = default;

        virtual void Reset() {
            table_.clear();
        }

        virtual bool IsTabu(const TKey& key) const {
            return table_.count(key);
        }

        virtual void Tabu(const TKey& key, Int tenure) {
            table_[key] = tenure;
        }

        virtual void Untabu(const TKey& key) {
            table_.erase(key);
        }

        virtual void Update() {
            for (auto it = table_.begin(); it != table_.end();) {
                --it->second;
                if (it->second <= 0)
                    it = table_.erase(it);
                else
                    ++it;
            }
        }

    protected:
        std::unordered_map<TKey, Int> table_;
    };

    class CircularTabuTable {
    public:
        CircularTabuTable(size_t size) : table_(size, 0) {}

        virtual ~CircularTabuTable() = default;

        virtual void Reset() {
            table_ = std::vector<Int>(table_.size(), 0);
        }

        virtual bool IsTabu(Int key) const {
            return table_[key % table_.size()] > 0;
        }

        virtual void Tabu(Int key, Int tenure) {
            table_[key % table_.size()] = tenure;
        }

        virtual void Update() {
            for (Int i=0; i<table_.size(); ++i) {
                if (table_[i]>0)
                    --table_[i];
            }
        }

    protected:
        std::vector<Int> table_;
    };
}