#pragma once
#include "rlop/common/base_algorithm.h"
#include "rlop/common/utils.h"
#include "rlop/common/random.h"

namespace rlop {
    // Implements a root parallel version of the Monte Carlo Tree Search (MCTS) algorithm. This class
    // allows for simultaneous exploration of multiple start nodes in parallel, making it suitable for
    // environments where multiple simulations can be run concurrently.
    class RootParallelMCTS : public BaseAlgorithm {
    public:
        struct Node {
            double mean_reward = 0;
            Int num_visits = 0;
            Int num_children = 0; // The number of children nodes expanded.
            std::vector<Node*> children;
        };
        
        // Constructs a RootParallelMCTS instance with specified parameters.
        //
        // Parameters:
        //   num_envs: The number of environments to run in parallel. Each environment represents an
        //             independent instance of the problem space for parallel exploration.
        //   coef: The exploration coefficient used in the UCB1 formula, Default is sqrt(2).
        RootParallelMCTS(Int num_envs, double coef = std::sqrt(2)) : 
            num_iters_(num_envs, 0),
            max_num_iters_(num_envs, 0),
            paths_(num_envs),
            rands_(num_envs),
            coef_(coef) 
        {}

        virtual ~RootParallelMCTS() = default;

        // Pure virtual function to return the total number of child states from the current state for a specified environment.
        virtual Int NumChildStates(Int env_i) const = 0;

        // Pure virtual function to determine whether a node has been fully expanded for a specified environment.
        //
        // Parameters:
        //   env_i: The index of environment.
        //
        // Return:
        //   bool: Returns true if the node is fully expanded.
        virtual bool IsExpanded(Int env_i, const Node& node) const = 0;

        // Pure virtual function to revert the state of a specified environment to its state at the beginning of the search. 
        //
        // Parameters:
        //   env_i: The index of environment.
        virtual void RevertState(Int env_i) = 0;

        // Pure virtual function to advance a specified environment state based on the selected child index. 
        //
        // Parameters:
        //   env_i: The index of environment.
        //   child_i: The index of the child to move to.
        //
        // Return:
        //   bool: Returns true if the step was successful. Returns false if the step was unsuccessful.
        virtual bool Step(Int env_i, Int child_i) = 0;

        // Pure virtual function to return the reward of the current state for a specified environment.
        virtual double Reward(Int env_i) = 0;

        // Resets the algorithm.
        virtual void Reset() override {
            for (Int i=0; i<paths_.size(); ++i) {
                if (!paths_[i].empty()) {
                    Release(paths_[i][0]);
                    delete paths_[i][0];
                }
                paths_[i] = { NewNode() };
            }
        }

        virtual void Reset(const std::vector<uint64_t>& seeds) {
            Reset();
            if (seeds.empty())
                return;
            for (Int i=0; i<rands_.size(); ++i) {
                if (i < seeds.size())
                    rands_[i].Seed(seeds[i]); 
                else
                    rands_[i].Seed(seeds.back()); 
            }
        }

        // Starts the asynchronous search across all environments, running each environment's search in parallel.
        virtual void SearchAsync(Int max_num_iters) {
            #pragma omp parallel for
            for (Int i=0; i<num_envs(); ++i) {
                Search(i, max_num_iters);
            }
        }

        // Performs the MCTS search over a maximum number of iterations for a specified environment .
        //
        // Parameters:
        //   env_i: The index of environment.
        //   max_num_iters: The maximum number of iterations.
        virtual void Search(Int env_i, Int max_num_iters) {
            num_iters_[env_i] = 0;
            max_num_iters_[env_i] = max_num_iters;
            while (Proceed(env_i)) {
                RevertState(env_i);
                if (Select(env_i) && Expand(env_i))
                    Simulate(env_i);
                BackPropagate(env_i);
                Update(env_i);
            }
        }

        // Checks if the search of a specified environment should continue.
        virtual bool Proceed(Int env_i) {
            return num_iters_[env_i] < max_num_iters_[env_i]; 
        }

        // Selects the next node to explore in the tree based on the tree policy for a specified environment.
        //
        // Parameters:
        //   env_i: The index of environment.
        virtual bool Select(Int env_i) {
            if (paths_[env_i].empty())
                return true;
            while (IsExpanded(env_i, *paths_[env_i].back())) {
                auto child_i = SelectTreePolicy(env_i);
                if (!child_i)
                    return false;
                paths_[env_i].push_back(paths_[env_i].back()->children[*child_i]); 
                if (!Step(env_i, *child_i)) 
                    return false;
            }
            return true;
        }

        // Expands the current node by adding a new child node to the tree for a specified environment.
        //
        // Parameters:
        //   env_i: The index of environment.
        virtual bool Expand(Int env_i) {
            if (paths_[env_i].back()->children.empty())
                paths_[env_i].back()->children = std::vector<Node*>(NumChildStates(env_i));
            if (paths_[env_i].back()->children.empty())
                return false;
            auto child_i = SelectToExpand(env_i);
            if (!child_i)
                return false;
            if (paths_[env_i].back()->children[*child_i] == nullptr) {
                paths_[env_i].back()->children[*child_i] = NewNode();
                ++paths_[env_i].back()->num_children;
            }
            paths_[env_i].push_back(paths_[env_i].back()->children[*child_i]);
            return Step(env_i, *child_i);
        }

        // Simulates the outcome from the current state to the end of the episode for a specified environment.
        //
        // Parameters:
        //   env_i: The index of environment.
        virtual bool Simulate(Int env_i) {
            while (true) {
                auto i = SelectRandom(env_i); 
                if (!i || !Step(env_i, *i))
                    return false;
            }
        }

        // Backpropagates the simulation results through the path in the tree for a specified environment.
        //
        // Parameters:
        //   env_i: The index of environment.
        virtual void BackPropagate(Int env_i) {
            double reward = Reward(env_i);
            while (paths_[env_i].size() > 1) {
                UpdateNode(env_i, reward);
                paths_[env_i].pop_back();
            }
            UpdateNode(env_i, reward);
        }

        virtual void Update(Int env_i) {
            ++num_iters_[env_i];
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

        virtual void UpdateNode(Int env_i, double reward) const {
            Node* node = paths_[env_i].back();
            node->mean_reward = (node->num_visits * node->mean_reward + reward) / (node->num_visits + 1.0);
            node->num_visits += 1;
        }

        // Computes the value of a child node using the UCB1 formula for a specified environment.
        //
        // Parameters:
        //   env_i: The index of environment.
        //   child_i: The index of the child node.
        //
        // Return:
        //   double: the value of the child node.
        virtual double TreePolicy(Int env_i, Int child_i) {
            if (paths_[env_i].back()->children[child_i] == nullptr)
                return std::numeric_limits<double>::lowest();
            return UCB1(paths_[env_i].back()->children[child_i]->mean_reward, paths_[env_i].back()->children[child_i]->num_visits, paths_[env_i].back()->num_visits, coef_);
        }

        // Selects the next child node to explore based on the tree policy for a specified environment.
        //
        // Parameters:
        //   env_i: The index of environment.
        //   child_i: The index of the child node.
        //
        // Returns:
        //   std::optional<Int>: The index of the child node with the highest UCB1 score. If the current node has no legal
        //                       children, returns std::nullopt.
        virtual std::optional<Int> SelectTreePolicy(Int env_i) {
            Int best = kIntNull;
            double best_score = std::numeric_limits<double>::lowest();
            for (Int i=0; i<paths_[env_i].back()->children.size(); ++i) {
                double score = TreePolicy(env_i, i);
                if (score > best_score) {
                    best = i;
                    best_score = score;
                }
            }
            if (best == kIntNull)
                return std::nullopt;
            return { best };
        }

        // Selects a child node to expand next from the current node's children for a specified environment. This selection can
        // be random or based on some heuristic.
        //
        // Parameters:
        //   env_i: The index of environment.
        //
        // Returns:
        //   std::optional<Int>: The index of the child node selected for expansion. If the current node has no legal children,
        //                       returns std::nullopt.
        virtual std::optional<Int> SelectToExpand(Int env_i) {
            if (paths_[env_i].back()->children.empty())
                return std::nullopt;
            return { rands_[env_i].Uniform(size_t(0), paths_[env_i].back()->children.size()-1) };
        }

        // Selects a child state randomly from the current state for a specified environment. This method is used during the 
        // simulation phase.
        //
        // Parameters:
        //   env_i: The index of environment.
        //
        // Returns:
        //   std::optional<Int>: The index of the randomly selected child node. If there is no legal child state available from
        //                       the current state, returns std::nullopt.
        virtual std::optional<Int> SelectRandom(Int env_i) {
            Int num_children = NumChildStates(env_i);
            if (num_children <= 0)
                return std::nullopt;
            return { rands_[env_i].Uniform(Int(0), num_children - 1) };
        }

        const Int num_envs() const {
            return paths_.size();
        }

        const std::vector<std::vector<Node*>>& paths() const {
            return paths_;
        }

        double coef() const {
            return coef_;
        }

        void set_coef(double coef) {
            coef_ = coef;
        }

    protected:
        double coef_;
        std::vector<Int> num_iters_;
        std::vector<Int> max_num_iters_;
        std::vector<std::vector<Node*>> paths_;
        std::vector<Random> rands_;
    };
}
