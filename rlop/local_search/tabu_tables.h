#pragma once
#include "rlop/common/utils.h"

namespace rlop {
    template<typename TKey>
    class HashTabuTable {
    public:
        virtual ~HashTabuTable() = default;

        virtual void Reset() {
            map_.clear();
        }

        virtual bool IsTabu(const TKey& key) const {
            return map_.count(key);
        }

        virtual void Tabu(const TKey& key, Int tenure) {
            map_[key] = tenure;
        }

        virtual void Untabu(const TKey& key) {
            map_.erase(key);
        }

        virtual void Update() {
            for (auto it = map_.begin(); it != map_.end();) {
                --it->second;
                if (it->second <= 0)
                    it = map_.erase(it);
                else
                    ++it;
            }
        }

        const std::unordered_map<TKey, Int>& map() const {
            return map_;
        } 

    protected:
        std::unordered_map<TKey, Int> map_;
    };

    class CircularTabuTable {
    public:
        CircularTabuTable(size_t size) : vec_(size, 0) {}

        virtual ~CircularTabuTable() = default;

        virtual void Reset() {
            vec_ = std::vector<Int>(vec_.size(), 0);
        }

        virtual bool IsTabu(Int key) const {
            return vec_[key % vec_.size()] > 0;
        }

        virtual void Tabu(Int key, Int tenure) {
            vec_[key % vec_.size()] = tenure;
        }

        virtual void Update() {
            for (Int i=0; i<vec_.size(); ++i) {
                if (vec_[i]>0)
                    --vec_[i];
            }
        }

        const std::vector<Int>& vec() const {
            return vec_;
        } 

    protected:
        std::vector<Int> vec_;
    };
}