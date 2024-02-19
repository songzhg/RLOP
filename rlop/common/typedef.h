#pragma once
#include "lib.h"

namespace rlop {
    using Int = int64_t;
    constexpr auto kIntNull = (std::numeric_limits<Int>::lowest)(); 
    constexpr auto kIntFull = (std::numeric_limits<Int>::max)(); 

#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                 \
    TypeName(const TypeName&) = delete;                                    \
    void operator=(const TypeName&) = delete;
}

