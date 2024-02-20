#include "problems/vrp/insertion_solver.h"
#include "local_search.h"
#include "tabu_search.h"
#include "simulate_annealing.h"
#include "rlop/common/timer.h"

int main() {
    using namespace vrp;

    rlop::Timer timer;
    rlop::Random rand;

    Int num_vehicles = 5;
    Int num_tasks = 30;

    std::vector<std::vector<Int>> matrix(num_tasks + num_vehicles, std::vector<Int>(num_tasks + num_vehicles, 0));
    for (Int i=0; i<matrix.size(); ++i) {
        for (Int j=0; j<matrix[i].size(); ++j) {
            if (i==j)
                continue;
            matrix[i][j] = rand.Uniform(1, 100); 
        }
    }

    auto get_cost = [&matrix](Int i, Int j){ 
        return matrix[i][j]; 
    };

    Routes routes(num_vehicles, num_tasks);
    routes.Reset();
    ArcCostManager manager(routes, get_cost);
    manager.Reset();
    OperatorSpace space(routes);
    space.Reset();
    Problem problem(&routes, &space, { &manager });

    InsertionSolver insertion(&problem);
    timer.Restart();
    insertion.Solve();
    timer.Stop();
    std::cout << "insertion: " << std::endl;
    routes.Print();
    std::cout << "total cost: " << problem.GetTotalCost() << std::endl;
    std::cout << "computing time: " << timer.duration() << "ms" << std::endl;
    std::cout << std::endl;
    

    LocalSearch local_search(get_cost);
    local_search.Reset(routes);
    timer.Restart();
    local_search.Search(10000);
    timer.Stop();
    std::cout << "local search: " << std::endl;
    local_search.best_routes().Print();
    std::cout << "total cost: " << local_search.best_cost() << std::endl;
    std::cout << "computing time: " << timer.duration() << "ms" << std::endl;    
    std::cout << std::endl;

    manager.Reset();
    space.Reset();
    TabuSearch tabu_search(get_cost);
    tabu_search.Reset(routes);
    timer.Restart();
    tabu_search.Search(10000);
    timer.Stop();
    std::cout << "tabu search: " << std::endl;
    tabu_search.best_routes().Print();
    std::cout << "total cost: " << tabu_search.best_cost() << std::endl;
    std::cout << "computing time: " << timer.duration() << "ms" << std::endl;    
    std::cout << std::endl;

    SimulatedAnnealing simulated_annealing(get_cost);
    simulated_annealing.Reset(routes);
    timer.Restart();
    simulated_annealing.Search(10000);
    timer.Stop();
    std::cout << "simulated annealing: " << std::endl;
    simulated_annealing.best_routes().Print();
    std::cout << "total cost: " << simulated_annealing.best_cost() << std::endl;
    std::cout << "computing time: " << timer.duration() << "ms" << std::endl;    
    std::cout << std::endl;
    
    return 0;
}