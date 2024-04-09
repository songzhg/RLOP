#pragma once
#include "routes.h"

namespace vrp {
    class CostManager {
    public:
        CostManager() = default;

        virtual ~CostManager() = default;

        virtual void Reset() = 0;

        virtual Int EvaluateDelta(const Operator& op) const = 0;
        
        virtual void Step(const Operator& op) = 0;

        virtual void Undo(const Operator& op) {}

        Int total_cost() const {
            return total_cost_;
        }

    protected:
        Int total_cost_ = 0; 
    };

    class ArcCostManager : public CostManager {
    public:
        ArcCostManager() = default;
        
        ArcCostManager(const Routes& routes, const std::function<Int(Int, Int)>& get_cost) : routes_(&routes), get_cost_(get_cost) {}
        
        virtual ~ArcCostManager() = default;

        virtual void Reset() override {
            total_cost_ = ComputeTotalCost(*routes_, get_cost_);    
        }
        
        virtual void Reset(const Routes& routes) {
            routes_ = &routes;
            Reset();
        }

        static Int ComputeTotalCost(const Routes& routes, const std::function<Int(Int, Int)>& get_cost) {
            Int total_cost = 0;    
            for (Int ri=0; ri<routes.num_routes(); ++ri) {
                total_cost += get_cost(routes.GetSentinel(ri), routes.GetStart(ri));
                for (Int ni=routes.GetStart(ri); ni!=routes.GetSentinel(ri); ni=routes.GetNext(ni)) {
                    total_cost += get_cost(ni, routes.GetNext(ni));
                }
            }
            return total_cost;
        }

        virtual Int EvaluateDelta(const Operator& op) const { 
            if (op.GetType() == Operator::Type::kInsertion)
                return EvaluateDelta(static_cast<const Insertion&>(op));        
            else if (op.GetType() == Operator::Type::kSwap)
                return EvaluateDelta(static_cast<const Swapping&>(op));
            else if (op.GetType() == Operator::Type::kMoving)
                return EvaluateDelta(static_cast<const Moving&>(op));    
            else if (op.GetType() == Operator::Type::kTwoOpt)
                return EvaluateDelta(static_cast<const TwoOpting&>(op));
            return 0;
        }

        virtual Int EvaluateDelta(const Insertion& insert) const {
            if (!routes_->IsInsertable(insert.node(), insert.to_node()))
                return std::numeric_limits<Int>::max();
            Int last = routes_->GetLast(insert.to_node());
            Int cost = 0;
            cost -= get_cost_(last, insert.to_node());
            cost += get_cost_(last, insert.node());
            cost += get_cost_(insert.node(), insert.to_node());
            return cost;
        }

        virtual Int EvaluateDelta(const Swapping& swap) const {
            if (!routes_->IsSwappable(swap.from_node(), swap.to_node()))
                return std::numeric_limits<Int>::max();
            Int last1 = routes_->GetLast(swap.from_node());
            Int next1 = routes_->GetNext(swap.from_node());
            Int last2 = routes_->GetLast(swap.to_node());
            Int next2 = routes_->GetNext(swap.to_node());
            Int cost = 0;
            cost += get_cost_(last2, swap.from_node());
            cost += get_cost_(swap.from_node(), next2);
            cost += get_cost_(last1, swap.to_node());
            cost += get_cost_(swap.to_node(), next1);
            cost -= get_cost_(last1, swap.from_node());
            cost -= get_cost_(swap.from_node(), next1);
            cost -= get_cost_(last2, swap.to_node());
            cost -= get_cost_(swap.to_node(), next2);
            return cost;
        }

        virtual Int EvaluateDelta(const Moving& move) const {
            if (!routes_->IsMovable(move.from_node(), move.to_node()))
                return std::numeric_limits<Int>::max();
            Int node = routes_->GetLast(move.from_node());
            Int last1 = routes_->GetLast(node);
            Int last2 = routes_->GetLast(move.to_node());
            Int cost = 0;
            cost += get_cost_(last1, move.from_node());
            cost += get_cost_(last2, node);
            cost += get_cost_(node, move.to_node());
            cost -= get_cost_(last1, node);
            cost -= get_cost_(node, move.from_node());
            cost -= get_cost_(last2, move.to_node());
            return cost;
        }

        virtual Int EvaluateDelta(const TwoOpting& two_opt) const {
            if (!routes_->IsTwoOptable(two_opt.from_node(), two_opt.to_node()))
                return std::numeric_limits<Int>::max();
            Int from_last = routes_->GetLast(two_opt.from_node());
            Int to_next = routes_->GetNext(two_opt.to_node());
            Int cost = 0;
            for (Int node=two_opt.from_node(); node!=to_next; node=routes_->GetNext(node)) {
                cost -= get_cost_(routes_->GetLast(node), node);
                if (node != to_next)
                    cost += get_cost_(routes_->GetNext(node), node); 
            }
            cost -= get_cost_(two_opt.to_node(), to_next); 
            cost += get_cost_(two_opt.from_node(), to_next); 
            cost += get_cost_(from_last, two_opt.to_node()); 
            return cost;
        }

        virtual void Step(const Operator& op) override {
            if (op.GetType() == Operator::Type::kInsertion)
                Step(static_cast<const Insertion&>(op));
            else if (op.GetType() == Operator::Type::kSwap)
                Step(static_cast<const Swapping&>(op));
            else if (op.GetType() == Operator::Type::kMoving)
                Step(static_cast<const Moving&>(op));
            else if (op.GetType() == Operator::Type::kTwoOpt)
                Step(static_cast<const TwoOpting&>(op));
        }

        virtual void Undo(const Operator& op) {
            if (op.GetType() == Operator::Type::kInsertion)
                Undo(static_cast<const Insertion&>(op));    
            else if (op.GetType() == Operator::Type::kSwap)
                Undo(static_cast<const Swapping&>(op));    
            else if (op.GetType() == Operator::Type::kMoving)
                Undo(static_cast<const Moving&>(op));    
            else if (op.GetType() == Operator::Type::kTwoOpt)
                Undo(static_cast<const TwoOpting&>(op));    
        }

        virtual void Step(const Insertion& insert) {
            Int last = routes_->GetLast(insert.node());
            total_cost_ -= get_cost_(last, insert.to_node());
            total_cost_ += get_cost_(last, insert.node());
            total_cost_ += get_cost_(insert.node(), insert.to_node());
        }

        virtual void Undo(const Insertion& insert) {
            Int last = routes_->GetLast(insert.node());
            Int next = routes_->GetNext(insert.node());
            total_cost_ += get_cost_(last, next);
            total_cost_ -= get_cost_(last, insert.node());
            total_cost_ -= get_cost_(insert.node(), next);
        }

        virtual void Step(const Swapping& swap) {
            Int last1 = routes_->GetLast(swap.from_node());
            Int next1 = routes_->GetNext(swap.from_node());
            Int last2 = routes_->GetLast(swap.to_node());
            Int next2 = routes_->GetNext(swap.to_node());
            total_cost_ += get_cost_(last1, swap.from_node());
            total_cost_ += get_cost_(swap.from_node(), next1);
            total_cost_ += get_cost_(last2, swap.to_node());
            total_cost_ += get_cost_(swap.to_node(), next2);
            total_cost_ -= get_cost_(last2, swap.from_node());
            total_cost_ -= get_cost_(swap.from_node(), next2);
            total_cost_ -= get_cost_(last1, swap.to_node());
            total_cost_ -= get_cost_(swap.to_node(), next1); 
        }

        virtual void Undo(const Swapping& swap) {
            Int last1 = routes_->GetLast(swap.from_node());
            Int next1 = routes_->GetNext(swap.from_node());
            Int last2 = routes_->GetLast(swap.to_node());
            Int next2 = routes_->GetNext(swap.to_node());
            total_cost_ -= get_cost_(last1, swap.from_node());
            total_cost_ -= get_cost_(swap.from_node(), next1);
            total_cost_ -= get_cost_(last2, swap.to_node());
            total_cost_ -= get_cost_(swap.to_node(), next2);
            total_cost_ += get_cost_(last2, swap.from_node());
            total_cost_ += get_cost_(swap.from_node(), next2);
            total_cost_ += get_cost_(last1, swap.to_node());
            total_cost_ += get_cost_(swap.to_node(), next1);
        }

        virtual void Step(const Moving& move) {
            Int last1 = routes_->GetLast(move.from_node());
            Int node = routes_->GetLast(move.to_node());
            Int last2 = routes_->GetLast(node);
            total_cost_ += get_cost_(last1, move.from_node());
            total_cost_ += get_cost_(last2, node);
            total_cost_ += get_cost_(node, move.to_node());
            total_cost_ -= get_cost_(last1, node);
            total_cost_ -= get_cost_(node, move.from_node());
            total_cost_ -= get_cost_(last2, move.to_node());
        }

        virtual void Undo(const Moving& move) {
            Int last1 = routes_->GetLast(move.from_node());
            Int node = routes_->GetLast(move.to_node());
            Int last2 = routes_->GetLast(node);
            total_cost_ -= get_cost_(last1, move.from_node());
            total_cost_ -= get_cost_(last2, node);
            total_cost_ -= get_cost_(node, move.to_node());
            total_cost_ += get_cost_(last1, node);
            total_cost_ += get_cost_(node, move.from_node());
            total_cost_ += get_cost_(last2, move.to_node());
        }

        virtual void Step(const TwoOpting& two_opt) {
            Int from_next = routes_->GetNext(two_opt.from_node());
            Int to_last = routes_->GetLast(two_opt.to_node());
            for (Int node=two_opt.to_node(); node!=from_next; node=routes_->GetNext(node)) {
                total_cost_ += get_cost_(routes_->GetLast(node), node);
                if (node != two_opt.from_node())
                    total_cost_ -= get_cost_(routes_->GetNext(node), node); 
            }
            total_cost_ += get_cost_(two_opt.from_node(), from_next); 
            total_cost_ -= get_cost_(to_last, two_opt.from_node()); 
            total_cost_ -= get_cost_(two_opt.to_node(), from_next); 
        }

        virtual void Undo(const TwoOpting& two_opt) {
            Int from_next = routes_->GetNext(two_opt.from_node());
            Int to_last = routes_->GetLast(two_opt.to_node());
            for (Int node=two_opt.to_node(); node!=from_next; node=routes_->GetNext(node)) {
                total_cost_ -= get_cost_(routes_->GetLast(node), node);
                if (node != two_opt.from_node())
                    total_cost_ += get_cost_(routes_->GetNext(node), node); 
            }
            total_cost_ -= get_cost_(two_opt.from_node(), from_next); 
            total_cost_ += get_cost_(to_last, two_opt.from_node()); 
            total_cost_ += get_cost_(two_opt.to_node(), from_next); 
        }

        const std::function<Int(Int, Int)>& get_cost() const {
            return get_cost_;
        }

    protected:
        const Routes* routes_ = nullptr;
        std::function<Int(Int, Int)> get_cost_;
    };
}