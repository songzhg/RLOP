#pragma once
#include "rlop/common/base_algorithm.h"
#include "rlop/common/utils.h"
#include "rlop/common/random.h"

namespace rlop {
    // Implements the Monte Carlo Tree Search (MCTS) algorithm for decision making in domains
    // with discrete action spaces.
    class MCTS : public BaseAlgorithm {
    public:
        struct Node {
            double mean_reward = 0;
            Int num_visits = 0;
            Int num_children = 0; // The number of children nodes expanded.
            std::vector<Node*> children;
        };
        
        // Constructs an MCTS with a exploration coefficient.
        //
        // Parameters:
        //   coef: The exploration coefficient used in the UCB1 formula, Default is sqrt(2).
        MCTS(double coef = std::sqrt(2)) : coef_(coef) {}

        virtual ~MCTS() = default;

        // Pure virtual function to return the total number of child states from the current state.
        virtual Int NumChildStates() const = 0;

        // Pure virtual function to determine whether a node has been fully expanded.
        virtual bool IsExpanded(const Node& node) const = 0;

        // Pure virtual function to revert the environment state to the state at the beginning of the search. 
        virtual void RevertState() = 0;

        // Pure virtual function to advance the environment state based on the selected child index. 
        //
        // Parameters:
        //   child_i: The index of the child to move to.
        //
        // Return:
        //   bool: Returns true if the step was successful.
        virtual bool Step(Int child_i) = 0;

        // Pure virtual function to return the reward of the current state. 
        virtual double Reward() = 0;

        // Resets the algorithm.
        virtual void Reset() override {
            if (!path_.empty()) {
                Release(path_[0]);
                delete path_[0];
            }
            path_ = { NewNode() };
        }

        virtual void SetSeed(uint64_t seed) {
            rand_.Seed(seed);
        }

        // Performs the MCTS search over a maximum number of iterations.
        //
        // Parameters:
        //   max_num_iters: The maximum number of iterations.
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

        // Checks if the search should continue.
        virtual bool Proceed() {
            return num_iters_ < max_num_iters_; 
        }

        // Selects the next node to explore in the tree based on the tree policy.
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

        // Expands the current node by adding a new child node to the tree.
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

        // Simulates the outcome from the current state to the end of the episode.
        virtual bool Simulate() {
            while (true) {
                auto i = SelectRandom(); 
                if (!i || !Step(*i))
                    return false;
            }
        }

        // Backpropagates the simulation results through the path in the tree.
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

        // Computes the value of a child node using the UCB1 formula.
        //
        // Parameters:
        //   child_i: The index of the child node.
        //
        // Return:
        //   double: the value of the child node.
        virtual double TreePolicy(Int child_i) {
            if (path_.back()->children[child_i] == nullptr)
                return std::numeric_limits<double>::lowest();
            return UCB1(path_.back()->children[child_i]->mean_reward, path_.back()->children[child_i]->num_visits, path_.back()->num_visits, coef_);
        }

        // Selects the next child node to explore based on the tree policy.
        //
        // Parameters:
        //   child_i: The index of the child node.
        //
        // Returns:
        //   std::optional<Int>: The index of the child node with the highest UCB1 score. If the current node has no
        //                       legal children, returns std::nullopt.
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

        // Selects a child node to expand next from the current node's children. This selection can be random or based on
        // some heuristic.
        // Returns:
        //   std::optional<Int>: The index of the child node selected for expansion. If the current node has no legal
        //                       children, returns std::nullopt.
        virtual std::optional<Int> SelectToExpand() {
            if (path_.back()->children.empty())
                return std::nullopt;
            return { rand_.Uniform(size_t(0), path_.back()->children.size() - 1) };
        }

        // Selects a child state randomly from the current state. This method is used during the simulation phase.
        //
        // Returns:
        //   std::optional<Int>: The index of the randomly selected child state. If there is no legal child state available,
        //                       returns std::nullopt.
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
