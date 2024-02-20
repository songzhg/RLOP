#pragma once
#include "utils.h"

namespace rlop {
    template <typename T>
    class CircularStack {
    public:
        CircularStack(size_t capacity = 1) : vec_(capacity) {}

        void Reset() {
            head_ = 0;
            pos_ = 0;
            full_ = false;
        }

        bool Empty() const {
            return !full_ && (head_ == pos_);
        }

        size_t Capacity() const {
            return vec_.size();
        }

        size_t Size() const {
            if (full_) {
                return vec_.size();
            }
            if (pos_ >= head_) {
                return pos_ - head_;
            }
            return vec_.size() + pos_ - head_;
        }

        void PushBack(const T& element) {
            vec_[pos_] = element;
            if (full_)
                head_ = (head_ + 1) % vec_.size();
            pos_ = (pos_ + 1) % vec_.size();
            full_ = head_ == pos_;
        }

        void PushBack(T&& element) {
            vec_[pos_] = std::move(element);
            if (full_)
                head_ = (head_ + 1) % vec_.size();
            pos_ = (pos_ + 1) % vec_.size();
            full_ = head_ == pos_;
        }

        void PopBack() {
            if (Empty())
                throw std::runtime_error("CircularBuffer: pop back on empty buffer.");
            full_ = false;
            pos_ = (pos_ == 0? vec_.size() : pos_) - 1;
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
            return vec_[(pos_ == 0? vec_.size() : pos_) - 1];
        } 

        const T& Back() const {
            if (Empty())
                throw std::runtime_error("CircularBuffer: pop back on empty buffer.");
            return vec_[(pos_ == 0? vec_.size() : pos_) - 1];
        }

        const std::vector<T>& vec() const {
            return vec_;
        }

        size_t head() const {
            return head_;
        }

        size_t pos() const {
            return pos_;
        }

        bool full() const {
            return full_;
        }
        
    private:
        std::vector<T> vec_;
        size_t head_ = 0;
        size_t pos_ = 0;
        bool full_ = false;
    };
}