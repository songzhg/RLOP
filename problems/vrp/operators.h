#pragma once
#include "rlop/common/typedef.h"

namespace vrp {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class Operator {
    public:
        enum class Type {
            kInsertion = 0,
            kSwap,
            kMoving,
            kTwoOpt,
        };

        virtual Type GetType() const = 0;
    };

    class Insertion : public Operator {
    public:
        Insertion(Int node, Int to_node) : node_(node), to_node_(to_node) {}

        Int node() const {
            return node_;
        }

        Int to_node() const {
            return to_node_;
        }

        virtual Type GetType() const override {
            return Type::kInsertion;
        }

    private:
        Int node_ = kIntNull;
        Int to_node_ = kIntNull;
    };

    class Swapping : public Operator {
    public:
        Swapping(Int from_node, Int to_node) : from_node_(from_node), to_node_(to_node) {}

        Int from_node() const {
            return from_node_;
        }

        Int to_node() const {
            return to_node_;
        }

        virtual Type GetType() const override {
            return Type::kSwap;
        }

    private:
        Int from_node_ = kIntNull;
        Int to_node_ = kIntNull;
    };

    class Moving : public Operator {
    public:
        Moving(Int from_node, Int to_node) : from_node_(from_node), to_node_(to_node) {}

        Int from_node() const {
            return from_node_;
        }

        Int to_node() const {
            return to_node_;
        }

        virtual Type GetType() const override {
            return Type::kMoving;
        }

    private:
        Int from_node_ = kIntNull;
        Int to_node_ = kIntNull;
    };

    class TwoOpting : public Operator {
    public:
        TwoOpting(Int from_node, Int to_node) : from_node_(from_node), to_node_(to_node) {}

        Int from_node() const {
            return from_node_;
        }

        Int to_node() const {
            return to_node_;
        }

        virtual Type GetType() const override {
            return Type::kTwoOpt;
        }

    private:
        Int from_node_ = kIntNull;
        Int to_node_ = kIntNull; 
    };
}