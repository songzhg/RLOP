#pragma once
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
            for (auto ptr : operators_) {
                delete ptr;
            }
            operators_.clear();
        }


        virtual void Reset(const Routes& routes) {
            routes_ = &routes;
            Reset();
        }

        virtual Int NumInsertions() const {
            return routes_->num_visited_nodes() < routes_->num_nodes()? routes_->num_visited_nodes() + routes_->num_routes() : 0; 
        }

        virtual std::unique_ptr<const Insertion> GetInsertion(Int i) const {
            if (i<routes_->num_visited_nodes())
                return std::make_unique<const Insertion>(routes_->num_visited_nodes(), i);
            Int route_i = i - routes_->num_visited_nodes();
            return std::make_unique<const Insertion>(routes_->num_visited_nodes(), routes_->GetSentinel(route_i));
        }

        virtual Int NumNeighbors() const {
            return operators_.size();
        }

        virtual const Operator* GetNeighbor(Int i) const {
            return operators_[i];
        }
        
        virtual void GenerateNeighbors() {
            Reset();
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

    protected: 
        const Routes* routes_ = nullptr;
        std::vector<Operator*> operators_;
    };
}