#pragma once
#include "rlop/common/random.h"
#include "operators.h"

namespace vrp {
    class OperatorSpace {
    public:
        OperatorSpace() = default;

        OperatorSpace(const Routes& routes) : routes_(&routes) {}

        virtual ~OperatorSpace() {
            for (auto ptr : operators_) {
                delete ptr;
            }
        }

        virtual void Reset() {
            ClearOperators();
        }


        virtual void Reset(const Routes& routes) {
            routes_ = &routes;
            Reset();
        }

        virtual void ClearOperators() {
            for (auto ptr : operators_) {
                delete ptr;
            }
            operators_.clear();
        }

        virtual void ClearInsertions() {
            for (auto ptr : insertions_) {
                delete ptr;
            }
            insertions_.clear();
        }

        virtual Int NumInsertions() const {
            return insertions_.size();
        }

        virtual const Insertion* GetInsertion(Int i) const {
            return insertions_[i];
        }

        virtual Int NumNeighbors() const {
            return operators_.size();
        }

        virtual const Operator* GetNeighbor(Int i) const {
            return operators_[i];
        }

        virtual void GenerateInsertions() {
            ClearInsertions();
            std::vector<Int> visited;
            std::vector<Int> unvisited;
            visited.reserve(routes_->num_nodes());
            unvisited.reserve(routes_->num_nodes());
            for (Int i=0; i<routes_->num_nodes(); ++i) {
                if (routes_->IsVisited(i))
                    visited.push_back(i);
                else
                    unvisited.push_back(i);
            }
            if (unvisited.empty())
                return;
            Int i = unvisited[rand_.Uniform(Int(0), (Int)unvisited.size()-1)];
            for (Int j : visited)
                insertions_.push_back(new Insertion(i, j));
            for (Int j=0; j<routes_->num_routes(); ++j)
                insertions_.push_back(new Insertion(i, routes_->GetSentinel(j)));
        }
        
        virtual void GenerateNeighbors() {
            ClearOperators();
            for (Int ri=0; ri<routes_->num_routes(); ++ri) {
                if (!routes_->IsStarted(ri))
                    continue;
                for (Int rj=0; rj<routes_->num_routes(); ++rj) {
                    if (ri == rj) {
                        for (Int ni=routes_->GetStart(ri); ni!=routes_->GetSentinel(ri); ni=routes_->GetNext(ni)) {
                            for (Int nj=routes_->GetNext(ni); nj!=routes_->GetSentinel(rj); nj=routes_->GetNext(nj)) {
                                if (nj != routes_->GetNext(ni))
                                    operators_.push_back(new Swapping(ni, nj));
                                operators_.push_back(new TwoOpting(ni, nj));
                            }
                        }
                    }
                    else {
                        for (Int ni=routes_->GetStart(ri); ni!=routes_->GetSentinel(ri); ni=routes_->GetNext(ni)) {
                            for (Int nj=routes_->GetStart(rj); nj!=routes_->GetSentinel(rj); nj=routes_->GetNext(nj)) {
                                operators_.push_back(new Swapping(ni, nj));
                            }
                        }
                    }
                    for (Int ni=routes_->GetStart(ri); ni!=routes_->GetSentinel(ri); ni=routes_->GetNext(ni)) {
                        for (Int nj=routes_->GetStart(rj); nj!=routes_->GetSentinel(rj); nj=routes_->GetNext(nj)) {
                            if (ni == nj || nj == routes_->GetLast(ni) || routes_->GetLast(ni) == routes_->GetSentinel(ri))
                                continue;
                            operators_.push_back(new Moving(ni, nj));
                        }
                    }
                }
            }
        }

        void Seed(uint64_t seed) {
            rand_.Seed(seed);
        }

    protected: 
        const Routes* routes_ = nullptr;
        std::vector<Operator*> operators_;
        std::vector<Insertion*> insertions_;
        rlop::Random rand_;               
    };
}