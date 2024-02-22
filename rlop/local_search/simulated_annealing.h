#pragma once
#include "local_search.h"
#include "rlop/common/random.h"

namespace rlop {
    // A template class for the Simulated Annealing optimization algorithm, extending LocalSearch.
    // This class template implements the simulated annealing heuristic for finding a near-optimal
    // solution by probabilistically accepting worse solutions to escape local optima. The algorithm
    // gradually reduces the temperature, decreasing the likelihood of accepting worse solutions
    // over time.
    //
    // Template parameters:
    //   TNeighbor: The type representing a neighbor solution.
    //   TCost: The cost type of a solution, defaulted to double. Must be an arithmetic type.
    template<typename TNeighbor, typename TCost = double>
    class SimulatedAnnealing : public LocalSearch<TNeighbor, TCost> {
    public:
        // Constructs a SimulatedAnnealing object with initial parameters for the annealing schedule.
        //
        // Parameters:
        //   initial_temp: The starting temperature.
        //   final_temp: The temperature at which the annealing process terminates.
        //   cooling_rate: The rate at which the temperature decreases in each iteration.
        SimulatedAnnealing(
            double initial_temp,
            double final_temp,
            double cooling_rate
        ) :
            initial_temp_(initial_temp),
            final_temp_(final_temp),
            cooling_rate_(cooling_rate)
        {}

        virtual ~SimulatedAnnealing() = default;

        // Pure virtual function to select a random neighbor.
        // Returns:
        //   std::optional<TNeighbor>: The neighbor being selected. If there is no legal neighbor,
        //                             returns std::nullopt.
        virtual std::optional<TNeighbor> SelectRandom() = 0;
        
        // Pure virtual function to select a local optimal neighbor.
        //
        // Returns:
        //   std::optional<TNeighbor>: The neighbor being selected. If there is no legal neighbor,
        //                             returns std::nullopt.
        virtual std::optional<TNeighbor> SelectLocal() = 0;

        // Pure virtual function to evaluate the cost of a neighbor.
        virtual TCost EvaluateNeighbor(const TNeighbor& neighbor) = 0;

        virtual void Reset() override {
            LocalSearch<TNeighbor, TCost>::Reset();
            temp_ = initial_temp_;
        }

        virtual void Reset(uint64_t seed) {
            Reset();
            rand_.Seed(seed);
        }

        virtual bool Proceed() override {
            if (!LocalSearch<TNeighbor, TCost>::Proceed())
                return false;
            return temp_ > final_temp_;
        }

        // Selects the next neighbor to consider for moving the search towards an optimal solution.
        // This method overrides the `Select` function defined in the base class, providing a specific
        // implementation strategy that first attempts to select a random neighbor. If a random neighbor
        // is not available or does not meet certain criteria, it then tries to select a local neighbor.
        //
        // Returns:
        //   std::optional<TNeighbor>: The neighbor being selected. If there is no legal neighbor,
        //                             returns std::nullopt.
        virtual std::optional<TNeighbor> Select() override {
            TCost cost = this->EvaluateSolution();
            auto neighbor = SelectRandom();
            if (!neighbor)
                return SelectLocal();
            TCost neighbor_cost = EvaluateNeighbor(*neighbor);
            if (Accept(neighbor_cost, cost)) 
                return neighbor;
            return SelectLocal();
        }
       
        // Determines whether to accept a new solution based on its cost relative to the current solution
        // and the current temperature.
        //
        // Parameters:
        //   new_cost: The cost of the new solution.
        //   cost: The cost of the current solution.
        virtual bool Accept(double new_cost, double cost) {
            if (new_cost < cost)
                return true;
            double prob = std::exp((cost - new_cost) / temp_);
            if (prob > rand_.Uniform(0.0, 1.0)) 
                return true;
            return false;
        }

        virtual void Update() override {
            temp_ *= (1.0 - cooling_rate_);
            ++this->num_iters_;
        }

        double temp() const {
            return temp_;
        }

        double initial_temp() const {
            return initial_temp_;
        }

        double final_temp() const {
            return final_temp_;
        }

        double cooling_rate() const {
            return cooling_rate_;
        }

    protected:
        double temp_;
        double initial_temp_;
        double final_temp_;
        double cooling_rate_;
        Random rand_;
    };
}