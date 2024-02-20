#pragma once
#include "cost_manager.h"
#include "operator_space.h"
 
namespace vrp {
    class Problem {
    public:
        Problem(Routes* routes, OperatorSpace* operator_space, const std::vector<CostManager*>& cost_managers) : 
            routes_(routes), 
            operator_space_(operator_space),
            cost_managers_(cost_managers) 
        {}

        virtual ~Problem() = default;

        virtual void Reset() {}

        virtual Int EvaluateDelta(const Operator& op) const { 
            Int delta = 0;
            for (auto manager : cost_managers_) {
                delta += manager->EvaluateDelta(op);
            }
            return delta;
        }

        virtual bool Step(const Operator& op) {
            if (!routes_->Step(op))
                return false;
            for (auto manager : cost_managers_) {
                manager->Step(op);
            }
            return true;
        }

        virtual void Undo(const Operator& op) {
            for (auto manager : cost_managers_) {
                manager->Undo(op);
            }
            routes_->Undo(op);    
        }

        virtual Int EncodeOperator(const Operator& op) const {
            if (op.GetType() == Operator::Type::kInsertion) {
                auto insert = static_cast<const Insertion&>(op);
                return insert.node() ^ insert.to_node() ^ 0;            
            }
            else if (op.GetType() == Operator::Type::kSwap) {
                auto swap = static_cast<const Swapping&>(op);
                return swap.from_node() ^ swap.to_node() ^ 1;
                
            }
            else if (op.GetType() == Operator::Type::kMoving) {
                auto move = static_cast<const Moving&>(op);
                return move.from_node() ^ move.to_node() ^ 2;    

            }
            else if (op.GetType() == Operator::Type::kTwoOpt) {
                auto two_opt = static_cast<const TwoOpting&>(op);
                return two_opt.from_node() ^ two_opt.to_node() ^ 3;
            }    
            return Int(0);
        }

        virtual Int GetTotalCost() const {
            Int total_cost = 0;
            for (auto manager : cost_managers_) {
                total_cost += manager->total_cost();
            }
            return total_cost;
        }

        Routes* routes() const {
            return routes_;
        }

        OperatorSpace* operator_space() const {
            return operator_space_;
        }

        const std::vector<CostManager*>& cost_managers() const {
            return cost_managers_;
        }

    protected:
        Routes* routes_ = nullptr;
        OperatorSpace* operator_space_ = nullptr;
        std::vector<CostManager*> cost_managers_;
    };
}