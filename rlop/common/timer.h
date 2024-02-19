#pragma once
#include <time.h>
#include <chrono>

namespace rlop {
    template<typename T = std::chrono::milliseconds>
    class Timer {
    public:
        Timer() { 
            Reset(); 
        }

        void Reset() {
            running_ = false;
            duration_ = 0;
        }

        void Start() {
            running_ = true;
            start_ = std::chrono::high_resolution_clock::now();
        }

        void Restart() {
            duration_ = 0;
            Start();
        }

        int64_t Stop() {
            if (running_) {
                auto const end = std::chrono::high_resolution_clock::now();
                duration_ += std::chrono::duration_cast<T> (end - start_).count();
                running_ = false;
            }
            return duration_;
        }

        int64_t duration() const {
            return duration_;
        } 

    private:
        bool running_;
        int64_t duration_ = 0;
        std::chrono::high_resolution_clock::time_point start_;
    };
}