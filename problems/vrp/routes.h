#pragma once
#include "operators.h"

namespace vrp {
    class Routes {
    public:
        Routes() = default;
        
        Routes(Int num_routes, Int num_nodes) : num_routes_(num_routes), num_nodes_(num_nodes) {}

        virtual ~Routes() = default;

        virtual void Reset() {
            num_visited_nodes_ = 0;
            node_to_route_ = std::vector<Int>(num_routes_ + num_nodes_, kIntNull);
            lasts_ = std::vector<Int>(num_routes_ + num_nodes_, kIntNull);
            nexts_ = std::vector<Int>(num_routes_ + num_nodes_, kIntNull);
            for (Int i=0; i<num_routes_; ++i) {
                lasts_[GetSentinel(i)] = GetSentinel(i);
                nexts_[GetSentinel(i)] = GetSentinel(i);
                node_to_route_[GetSentinel(i)] = i; 
            }
        }

        void Print() const {
            std::cout << "Routes:" << std::endl;
            for (Int ri = 0; ri < num_routes(); ++ri) {
                std::cout << "route " << ri << ": ";
                Int ni = GetStart(ri);
                while (ni != GetSentinel(ri)) {
                    std::cout << ni << " -> ";
                    ni = nexts_[ni];
                }
                std::cout << "end" << std::endl;
            }
            std::cout << std::endl;
        }

        virtual bool Step(const Operator& op) {
            if (op.GetType() == Operator::Type::kInsertion) {
                auto insert = static_cast<const Insertion&>(op);
                if (!Insert(insert.node(), insert.to_node()))
                    return false;
            }
            else if (op.GetType() == Operator::Type::kSwap) {
                auto swap = static_cast<const Swapping&>(op);
                if (!Swap(swap.from_node(), swap.to_node()))
                    return false;
            }
            else if (op.GetType() == Operator::Type::kMoving) {
                auto move = static_cast<const Moving&>(op);
                if (!Move(move.from_node(), move.to_node()))
                    return false;
            }
            else if (op.GetType() == Operator::Type::kTwoOpt) {
                auto opt = static_cast<const TwoOpting&>(op);
                if (!TwoOpt(opt.from_node(), opt.to_node()))
                    return false;
            }
            return true;
        }

        virtual void Undo(const Operator& op) {
            if (op.GetType() == Operator::Type::kInsertion) {
                auto insert = static_cast<const Insertion&>(op);
                Erase(insert.node());
            }
            else if (op.GetType() == Operator::Type::kSwap) {
                auto swap = static_cast<const Swapping&>(op);
                Swap(swap.from_node(), swap.to_node());
            }
            else if (op.GetType() == Operator::Type::kMoving) {
                auto move = static_cast<const Moving&>(op);
                Move(move.to_node(), move.from_node());
            }
            else if (op.GetType() == Operator::Type::kTwoOpt) {
                auto two_opt = static_cast<const TwoOpting&>(op);
                TwoOpt(two_opt.to_node(), two_opt.from_node()); 
            }
        }

        bool Erase(Int node) {
            if (!IsErasable(node)) 
                return false;
            Int route = GetRoute(node);
            nexts_[GetLast(node)] = GetNext(node);
            lasts_[GetNext(node)] = GetLast(node);
            lasts_[node] = kIntNull;
            nexts_[node] = kIntNull;
            node_to_route_[node] = kIntNull;
            --num_visited_nodes_;
            return true;
        }

        bool Insert(Int node, Int to_node) {
            if (!IsInsertable(node, to_node))
                return false;
            nexts_[node] = to_node;
            lasts_[node] = GetLast(to_node);
            if (GetLast(to_node) != kIntNull)
                nexts_[GetLast(to_node)] = node;
            lasts_[to_node] = node;
            node_to_route_[node] = GetRoute(to_node);
            ++num_visited_nodes_;
            return true;
        }

        bool Swap(Int from_node, Int to_node) {
            if (!IsSwappable(from_node, to_node)) 
                return false;
            nexts_[GetLast(from_node)] = to_node;
            nexts_[GetLast(to_node)] = from_node;
            lasts_[GetNext(from_node)] = to_node;
            lasts_[GetNext(to_node)] = from_node;
            std::swap(nexts_[from_node], nexts_[to_node]);
            std::swap(lasts_[from_node], lasts_[to_node]);
            std::swap(node_to_route_[from_node], node_to_route_[to_node]);
            return true;
        }

        bool Move(Int from_node, Int to_node) {
            if (!IsMovable(from_node, to_node)) 
                return false;
            Int moved = GetLast(from_node);
            nexts_[GetLast(moved)] = from_node;
            lasts_[from_node] = GetLast(moved);
            nexts_[GetLast(to_node)] = moved;
            lasts_[moved] = GetLast(to_node);
            nexts_[moved] = to_node;
            lasts_[to_node] = moved;
            node_to_route_[moved] = node_to_route_[to_node];
            return true;
        }

        bool TwoOpt(Int from_node, Int to_node) {
            if (!IsTwoOptable(from_node, to_node))
                return false;
            Int to_next = GetNext(to_node);
            Int from_last = GetLast(from_node);
            for (Int node=from_node; node!=to_next; node=GetLast(node)) {
                std::swap(nexts_[node], lasts_[node]);
            }
            nexts_[from_last] = to_node;
            lasts_[to_next] = from_node;
            nexts_[from_node] = to_next;
            lasts_[to_node] = from_last;
            return true;
        }

        virtual Int GetSentinel(Int route) const {
            return num_nodes_ + route;
        }

        virtual bool IsErasable(Int node) const {
            return IsVisited(node);
        }

        virtual bool IsInsertable(Int node, Int to_node) const {
            return !IsVisited(node) && IsVisited(to_node);
        }

        virtual bool IsSwappable(Int from_node, Int to_node) const {
            return IsVisited(from_node) && IsVisited(to_node) && GetLast(from_node) != to_node && GetNext(from_node) != to_node;
        }

        virtual bool IsMovable(Int from_node, Int to_node) const {
            return IsVisited(from_node) && IsVisited(to_node) && GetLast(from_node) != to_node && GetLast(from_node) < num_nodes();
        }

        virtual bool IsTwoOptable(Int from_node, Int to_node) const {
            return IsVisited(from_node) && IsVisited(to_node) && GetRoute(from_node) == GetRoute(to_node);
        }

        virtual bool IsStarted(Int route) const {
            return GetStart(route) != GetSentinel(route); 
        }

        virtual bool IsVisited(Int node) const {
            return node_to_route_[node] != kIntNull; 
        }

        virtual Int GetStart(Int route) const {
            return nexts_[GetSentinel(route)];
        }

        virtual Int GetCurrent(Int route) const { 
            return lasts_[GetSentinel(route)];
        }

        virtual Int GetLast(Int node) const { 
            return lasts_[node];
        }

        virtual Int GetNext(Int node) const { 
            return nexts_[node]; 
        }

        virtual Int GetRoute(Int node) const { 
            return node_to_route_[node]; 
        }

        Int num_routes() const { 
            return num_routes_; 
        };

        Int num_nodes() const { 
            return num_nodes_;
        }

        Int num_visited_nodes() const {
            return num_visited_nodes_;
        }

    protected:
        Int num_routes_ = 0;
        Int num_nodes_ = 0;
        Int num_visited_nodes_ = 0;
        std::vector<Int> node_to_route_;
        std::vector<Int> lasts_;
        std::vector<Int> nexts_;
    };
}
