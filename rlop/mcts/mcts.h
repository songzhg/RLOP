#pragma once
#include "rlop/common/base_algorithm.h"
#include "rlop/common/utils.h"
#include "rlop/common/random.h"

namespace rlop {
    class MCTS : public BaseAlgorithm {
    public:
        struct Node {
            double mean_reward = 0;
            Int num_visits = 0;
            Int num_children = 0;
            std::vector<Node*> children;
        };
        
        MCTS(double coef = std::sqrt(2)) : coef_(coef) {}

        virtual ~MCTS() = default;

        virtual Int NumChildStates() const = 0;

        virtual bool IsExpanded(const Node& node) const = 0;

        virtual void RevertState() = 0;

        virtual bool Step(Int child_i) = 0;

        virtual double Reward() = 0;

        virtual void Reset() override {
            if (!path_.empty()) {
                Release(path_[0]);
                delete path_[0];
            }
            path_ = { NewNode() };
        }

        virtual void Reset(uint64_t seed) {
            Reset();
            rand_.Seed(seed);
        }

        virtual void Search(Int max_num_iters) {
            num_iters_ = 0;
            max_num_iters_ = max_num_iters;
            while (Proceed()) {
                RevertState();
                if (Select() && Expand())
                    Simulate();
                BackPropagate();
                Update();
            }
        }

        virtual bool Proceed() {
            return num_iters_ < max_num_iters_; 
        }

        virtual bool Select() {
            if (path_.empty())
                return true;
            while (IsExpanded(*path_.back())) {
                auto child_i = SelectTreePolicy();
                if (!child_i)
                    return false;
                path_.push_back(path_.back()->children[*child_i]); 
                if (!Step(*child_i)) 
                    return false;
            }
            return true;
        }

        virtual bool Expand() {
            if (path_.back()->children.empty())
                path_.back()->children = std::vector<Node*>(NumChildStates());
            if (path_.back()->children.empty())
                return false;
            auto child_i = SelectToExpand();
            if (!child_i)
                return false;
            if (path_.back()->children[*child_i] == nullptr) {
                path_.back()->children[*child_i] = NewNode();
                ++path_.back()->num_children;
            }
            path_.push_back(path_.back()->children[*child_i]);
            return Step(*child_i);
        }

        virtual bool Simulate() {
            while (true) {
                auto i = SelectRandom(); 
                if (!i || !Step(*i))
                    return false;
            }
        }

        virtual void BackPropagate() {
            double reward = Reward();
            while (path_.size() > 1) {
                UpdateNode(reward);
                path_.pop_back();
            }
            UpdateNode(reward);
        }

        virtual void Update() {
            ++num_iters_;
        }

        virtual void Release(Node* node) {
            if (node == nullptr) 
                return;
            for (Int i=0; i<node->children.size(); ++i) {
                Node* child = node->children[i];
                Release(child);
                delete child;
            }
            node->children.clear();
        }

        virtual Node* NewNode() const {
            return new Node();
        }

        virtual void UpdateNode(double reward) const {
            Node* node = path_.back();
            node->mean_reward = (node->num_visits * node->mean_reward + reward) / (node->num_visits + 1.0);
            node->num_visits += 1;
        }

        virtual double TreePolicy(Int child_i) {
            if (path_.back()->children[child_i] == nullptr)
                return std::numeric_limits<double>::lowest();
            return UCB1(path_.back()->children[child_i]->mean_reward, path_.back()->children[child_i]->num_visits, path_.back()->num_visits, coef_);
        }

        virtual std::optional<Int> SelectTreePolicy() {
            Int best = kIntNull;
            double best_score = std::numeric_limits<double>::lowest();
            for (Int i=0; i<path_.back()->children.size(); ++i) {
                double score = TreePolicy(i);
                if (score > best_score) {
                    best = i;
                    best_score = score;
                }
            }
            if (best == kIntNull)
                return std::nullopt;
            return { best };
        }

        virtual std::optional<Int> SelectToExpand() {
            if (path_.back()->children.empty())
                return std::nullopt;
            return { rand_.Uniform(size_t(0), path_.back()->children.size() - 1) };
        }

        virtual std::optional<Int> SelectRandom() {
            Int num_children = NumChildStates();
            if (num_children <= 0)
                return std::nullopt;
            return { rand_.Uniform(Int(0), num_children - 1) };
        }

        const std::vector<Node*>& path() const {
            return path_;
        }

        double coef() const {
            return coef_;
        }

        void set_coef(double coef) {
            coef_ = coef;
        }

    protected:
        double coef_;
        Int num_iters_ = 0;
        Int max_num_iters_ = 0;
        std::vector<Node*> path_;
        Random rand_;
    };
}
