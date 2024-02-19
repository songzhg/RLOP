#pragma once
#include "typedef.h"

namespace rlop { 
    class BaseAlgorithm {
    public:
        BaseAlgorithm() = default;

        virtual ~BaseAlgorithm() = default;

        virtual void Reset() {}
    };
}