#pragma once
#include "utils.h"

namespace rlop {
    // A template class for a circular buffer that offers efficient FIFO operations, which is suitable for 
    // fixed-size buffers where old elements are overwritten by new ones once the capacity is reached.
    template <typename T>
    class CircularStack {
    public:
        CircularStack(size_t capacity = 1) : vec_(capacity) {}

        void Reset() {
            head_ = 0;
            tail_ = 0;
            full_ = false;
        }

        bool Empty() const {
            return !full_ && (head_ == tail_);
        }

        size_t Capacity() const {
            return vec_.size();
        }

        size_t Size() const {
            if (full_) {
                return vec_.size();
            }
            if (tail_ >= head_) {
                return tail_ - head_;
            }
            return vec_.size() + tail_ - head_;
        }

        void Push(const T& element) {
            vec_[tail_] = element;
            if (full_)
                head_ = (head_ + 1) % vec_.size();
            tail_ = (tail_ + 1) % vec_.size();
            full_ = head_ == tail_;
        }

        void Push(T&& element) {
            vec_[tail_] = std::move(element);
            if (full_)
                head_ = (head_ + 1) % vec_.size();
            tail_ = (tail_ + 1) % vec_.size();
            full_ = head_ == tail_;
        }

        void Pop() {
            if (Empty())
                throw std::runtime_error("CircularBuffer: pop back on empty buffer.");
            full_ = false;
            tail_ = (tail_ == 0? vec_.size() : tail_) - 1;
        }

        T& Front()  {
            if (Empty())
                throw std::runtime_error("CircularBuffer: get elements on empty buffer.");
            return vec_[head_];
        }

        const T& Front() const {
            if (Empty())
                throw std::runtime_error("CircularBuffer: get elements on empty buffer.");
            return vec_[head_];
        } 

        T& Back() {
            if (Empty())
                throw std::runtime_error("CircularBuffer: get elements on empty buffer.");
            return vec_[(tail_ == 0? vec_.size() : tail_) - 1];
        } 

        const T& Back() const {
            if (Empty())
                throw std::runtime_error("CircularBuffer: pop back on empty buffer.");
            return vec_[(tail_ == 0? vec_.size() : tail_) - 1];
        }

        const std::vector<T>& vec() const {
            return vec_;
        }

        size_t head() const {
            return head_;
        }

        size_t tail() const {
            return tail_;
        }

        bool full() const {
            return full_;
        }
        
    private:
        std::vector<T> vec_;
        size_t head_ = 0;
        size_t tail_ = 0;
        bool full_ = false;
    };
}