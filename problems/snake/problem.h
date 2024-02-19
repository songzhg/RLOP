#pragma once
#include "game.h"
#include "rlop/common/torch_utils.h"

namespace snake {
    class Problem {
    public:
        Problem(bool render = false) {
            if (render) {
                graphics_ = std::make_unique<Graphics>(engine_);
            }
        }

        virtual ~Problem() = default;

        virtual void Reset() {
            engine_.Reset();
            if (graphics_)
                graphics_->Reset();
        }

        virtual void Reset(uint64_t seed) {
            Reset();
            engine_.set_seed(seed); 
        }

        virtual void Reset(Engine&& engine) {
            engine_ = std::move(engine);
        }

        virtual void Reset(const Engine& engine) {
            engine_ = engine;
        }

        virtual torch::Tensor GetObservation() const {
            std::vector<float> food(grid_size(), 0); 
            for (const auto& pos : engine_.foods()) {
                food[pos.second*grid_width()+pos.first] = 1.0;
            }
            std::vector<torch::Tensor> channels;
            channels.reserve(5);
            channels.push_back(torch::tensor(food).view({-1, grid_width()}));
            for (const auto& snake : engine_.snakes()) {
                std::vector<float> head(grid_size(), 0);
                std::vector<float> body(grid_size(), 0);
                std::vector<float> tail(grid_size(), 0);
                std::vector<float> old_head(grid_size(), 0);
                if (snake.alive) {
                    head[snake.body.front().second*grid_width()+snake.body.front().first] = 1.0;
                    tail[snake.body.back().second*grid_width()+snake.body.back().first] = 1.0;
                    Int rev_dir = engine_.GetReverseDir(snake.dir);
                    auto pos = engine_.GetNextPos(snake.body.front(), rev_dir);
                    if (!engine_.OutOfBoundary(pos) && engine_.num_steps() != 0)
                        old_head[pos.second*grid_width()+pos.first] = 1.0; 
                    for (Int i=0; i<snake.body.size(); ++i) {
                        body[snake.body[i].second*grid_width()+snake.body[i].first] = 1.0 - 1.0 / (grid_size() + 1) * i;
                    }
                }
                channels.push_back(torch::tensor(head).view({-1, grid_width()}));
                channels.push_back(torch::tensor(old_head).view({-1, grid_width()}));
                channels.push_back(torch::tensor(body).view({-1, grid_width()}));
                channels.push_back(torch::tensor(tail).view({-1, grid_width()}));
            }
            return torch::stack(channels);
        }

        virtual Int NumActions() const {
            return 4;
        }

        virtual Int GetAction(Int i) const {
            return i;
        }

        virtual bool Step(const std::vector<Int>& actions) {
            for (Int i=0; i<actions.size(); ++i) {
                engine_.SetDir(i, actions[i]);
            }
            engine_.Update(); 
            return !engine_.IsEnd();
        }

        virtual bool Render() {
            if (graphics_) {
                if (!graphics_->IsOpen())
                    return false;
                graphics_->HandleEvents();
                graphics_->Render();
            }
            return true;
        }

        Int grid_width() const {
            return engine_.grid_width();
        }

        Int grid_height() const {
            return engine_.grid_height();
        }

        Int grid_size() const {
            return engine_.grid_size();
        }

        std::vector<Int> observation_sizes() const {
            return { 5, grid_height(), grid_width() };
        }

        std::vector<Int> action_sizes() const {
            return {};
        }

        Int max_num_steps() const {
            return engine_.max_num_steps();
        }

        const Engine& engine() const {
            return engine_;
        }

    protected:
        Engine engine_;
        std::unique_ptr<Graphics> graphics_ = nullptr;
    };

    class VectorProblem {
    public:
        VectorProblem(Int num_envs, bool render = false) : engines_(num_envs) {
            if (render) {
                std::vector<const Engine*> engine_ptrs;
                engine_ptrs.reserve(num_envs);
                for (Int i=0; i<num_envs; ++i) {
                    engine_ptrs.push_back(&engines_[i]);
                }
                graphics_ = std::make_unique<Graphics>(std::move(engine_ptrs));
                graphics_->Reset();
            }
        }

        virtual ~VectorProblem() = default; 

        virtual void Reset(Int env_i) {
            engines_[env_i].Reset();
        }

        virtual void Reset(Int env_i, uint64_t seed) {
            Reset(env_i);
            engines_[env_i].set_seed(seed); 
        }

        virtual void Reset(Int env_i, Engine&& engine) {
            engines_[env_i] = std::move(engine);
        }

        virtual void Reset(Int env_i, const Engine& engine) {
            engines_[env_i] = engine;
        }

        virtual torch::Tensor GetObservation(Int env_i) const {
            std::vector<float> food(grid_size(), 0); 
            for (const auto& pos : engines_[env_i].foods()) {
                food[pos.second*grid_width()+pos.first] = 1.0;
            }
            std::vector<torch::Tensor> channels;
            channels.reserve(5);
            channels.push_back(torch::tensor(food).view({-1, grid_width()}));
            for (const auto& snake : engines_[env_i].snakes()) {
                std::vector<float> head(grid_size(), 0);
                std::vector<float> body(grid_size(), 0);
                std::vector<float> tail(grid_size(), 0);
                std::vector<float> old_head(grid_size(), 0);
                if (snake.alive) {
                    head[snake.body.front().second*grid_width()+snake.body.front().first] = 1.0;
                    tail[snake.body.back().second*grid_width()+snake.body.back().first] = 1.0;
                    Int rev_dir = engines_[env_i].GetReverseDir(snake.dir);
                    auto pos = engines_[env_i].GetNextPos(snake.body.front(), rev_dir);
                    if (!engines_[env_i].OutOfBoundary(pos) && engines_[env_i].num_steps() != 0)
                        old_head[pos.second*grid_width()+pos.first] = 1.0; 
                    for (Int i=0; i<snake.body.size(); ++i) {
                        body[snake.body[i].second*grid_width()+snake.body[i].first] = 1.0 - 1.0 / (grid_size() + 1) * i;
                    }
                }
                channels.push_back(torch::tensor(head).view({-1, grid_width()}));
                channels.push_back(torch::tensor(old_head).view({-1, grid_width()}));
                channels.push_back(torch::tensor(body).view({-1, grid_width()}));
                channels.push_back(torch::tensor(tail).view({-1, grid_width()}));
            }
            return torch::stack(channels);
        }

        virtual Int NumActions() const {
            return 4;
        }

        virtual Int GetAction(Int i) const {
            return i;
        }

        virtual bool Step(Int env_i, const std::vector<Int>& actions) {
            for (Int i=0; i<actions.size(); ++i) {
                engines_[env_i].SetDir(i, actions[i]);
            }
            engines_[env_i].Update(); 
            return !engines_[env_i].IsEnd();
        }

        virtual bool Render() {
            if (graphics_) {
                if (!graphics_->IsOpen())
                    return false;
                graphics_->HandleEvents();
                graphics_->Render();
            }
            return true;
        }

        Int num_problems() const {
            return engines_.size();
        }

        Int grid_width() const {
            return engines_[0].grid_width();
        }

        Int grid_height() const {
            return engines_[0].grid_height();
        }

        Int grid_size() const {
            return engines_[0].grid_size();
        }

        std::vector<Int> observation_sizes() const {
            return { 5, grid_height(), grid_width() };
        }

        std::vector<Int> action_sizes() const {
            return {};
        }

        Int max_num_steps() const {
            return engines_[0].max_num_steps();
        }

        const std::vector<Engine>& engines() const {
            return engines_;
        }

    protected:
        std::vector<Engine> engines_;
        std::unique_ptr<Graphics> graphics_ = nullptr;
    };
}