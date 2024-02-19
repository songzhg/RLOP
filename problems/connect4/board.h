#pragma once
#include "rlop/common/lib.h"

namespace connect4 {
    class Board {
    public:
        using bitboard = uint64_t;

        static constexpr int kWidth_ = 7;
        static constexpr int kHeight_ = 6;
        static constexpr int kSize_ = kWidth_ * kHeight_;
        static constexpr int kH1_ = kHeight_ + 1; 
        static constexpr int kH2_ = kHeight_ + 2;
        static constexpr int kCol1_ = (((bitboard)1<<kH1_)-(bitboard)1);
        static constexpr bitboard kBottom = 0b0000001000000100000010000001000000100000010000001;

        /*
        .  .  .  .  .  .  .
        5 12 19 26 33 40 47
        4 11 18 25 32 39 46
        3 10 17 24 31 38 45
        2  9 16 23 30 37 44
        1  8 15 22 29 36 43
        0  7 14 21 28 35 42 
        */

        Board() = default;

        virtual ~Board() = default;

        virtual void Reset() {
            num_moves_ = 0;
            players_ = {};
            heights_ = {};
        }

        virtual void Reset(const std::string& position) {
            Reset();
            if (position.size() != kSize_)
                return;
            for (int i=0; i<position.size(); ++i) {
                if (position[i] == 'O') {
                    int row = kHeight_ -  i / kWidth_ - 1;
                    int col = i % kWidth_;
                    heights_[col] = std::max(heights_[col], static_cast<char>(row + 1));
                    players_[0] ^= static_cast<bitboard>(1) << (row + kH1_ * col);
                    ++num_moves_;
                }
                else if (position[i] == 'X') {
                    int row = kHeight_ -  i / kWidth_ - 1;
                    int col = i % kWidth_;
                    heights_[col] = std::max(heights_[col], static_cast<char>(row + 1));
                    players_[1] ^= static_cast<bitboard>(1) << (row + kH1_ * col);
                    ++num_moves_;
                }
            }
        }

        virtual void Print() const {
            for (int row = kHeight_ - 1; row >= 0; row--) {
                for (int col = 0; col < kWidth_; col++) {
                    bitboard mask = static_cast<bitboard>(1) << (row + kH1_ * col);
                    if (players_[0] & mask)
                        std::cout << "O ";
                    else if (players_[1] & mask)
                        std::cout << "X ";
                    else
                        std::cout << ". ";
                }
                std::cout << std::endl;
            }
            for (int col = 0; col < kWidth_; col++) {
                std::cout << col << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }

        virtual bool IsOver() const {
            return Win() || IsFull();
        }

        virtual bool IsFull() const {
            return num_moves_ >= kSize_;
        }

        virtual bool IsPlayable(int col) const {
            return heights_[col] < kHeight_;
        }

        virtual bool Win() const {
            int current = 1 - num_moves_ % 2;
            bitboard h = players_[current] & (players_[current] >> kH1_);
            bitboard v = players_[current] & (players_[current] >> 1);
            bitboard d1 = players_[current] & (players_[current] >> kHeight_) ;
            bitboard d2 = players_[current] & (players_[current] >> kH2_);
            return (h & (h >> (2 * kH1_))) ||
                (v  & (v >> 2)) ||
                (d1 & (d1 >> (2 * kHeight_))) || 
                (d2 & (d2 >> (2 * kH2_)));
        }

        virtual bitboard PositionEncode() const {
            return players_[num_moves_ % 2] + players_[0] + players_[1] + kBottom;
        }

        virtual bool MakeMove(int col) {
            if (col < 0 || col >= kWidth_ || heights_[col] >= kHeight_)
                return false;
            players_[num_moves_ % 2] ^= static_cast<bitboard>(1) << (heights_[col]++ + kH1_ * col);
            ++num_moves_;
            return true;
        }
        
        virtual void UndoMove(int col) {
            --num_moves_;
            players_[num_moves_ % 2] ^= static_cast<bitboard>(1) << (--heights_[col] + kH1_ * col);
        }

        int num_moves() const {
            return num_moves_;
        }

        const std::array<bitboard, 2>& players() const {
            return players_;
        }

        const std::array<char, kWidth_>& heights() const {
            return heights_;
        }

    protected:
        std::array<bitboard, 2> players_;
        std::array<char, kWidth_> heights_;
        int num_moves_ = 0;
    };
}