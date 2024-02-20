#pragma once
#include "local_search.h"
#include "rlop/common/random.h"

namespace rlop {
    template<typename TNeighbor, typename TCost = double>
    class SimulatedAnnealing : public LocalSearch<TNeighbor, TCost> {
    public:
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

        virtual std::optional<TNeighbor> SelectRandom() = 0;
        
        virtual std::optional<TNeighbor> SelectLocal() = 0;

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