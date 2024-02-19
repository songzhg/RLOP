#include "problems/vrp/insertion_solver.h"
#include "local_search.h"
#include "tabu_search.h"
#include "simulate_annealing.h"
#include "rlop/common/timer.h"

int main() {
	using namespace vrp;

	rlop::Timer timer;
	rlop::Random rand;

	Int num_vehicles = 10;
	Int num_tasks = 200;

	double insert_duration = 0;
	double local_duration = 0;
	double tabu_duration = 0;
	double anneal_duration = 0;

	double insert_cost = 0;
	double local_cost = 0;
	double tabu_cost = 0;
	double anneal_cost = 0;

	for (Int t = 0; t < 100; ++t) {
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
		insert_duration += timer.Stop();
		insert_cost += problem.GetTotalCost();
		// std::cout << "insertion: " << std::endl;
		// routes.Print();
		// std::cout << "total cost: " << problem.GetTotalCost() << std::endl;
		// std::cout << "computing time: " << timer.duration() << "ms" << std::endl;
		// std::cout << std::endl;
		

		Routes initial_routes = routes;

		LocalSearch local_search(&problem);
		timer.Restart();
		local_search.Search(10000);
		local_duration += timer.Stop();
		local_cost += local_search.best_cost();
		// std::cout << "local search: " << std::endl;
		// local_search.best_routes().Print();
		// std::cout << "total cost: " << local_search.best_cost() << std::endl;
		// std::cout << "computing time: " << timer.duration() << "ms" << std::endl;	
		// std::cout << std::endl;

		routes = initial_routes;
		manager.Reset();
		space.Reset();
		TabuSearch tabu_search(&problem);
		tabu_search.Reset();
		timer.Restart();
		tabu_search.Search(10000);
		tabu_duration += timer.Stop();
		tabu_cost += tabu_search.best_cost();
		// std::cout << "tabu search: " << std::endl;
		// tabu_search.best_routes().Print();
		// std::cout << "total cost: " << tabu_search.best_cost() << std::endl;
		// std::cout << "computing time: " << timer.duration() << "ms" << std::endl;	
		// std::cout << std::endl;

		routes = initial_routes;
		manager.Reset();
		space.Reset();
		SimulatedAnnealing simulated_annealing(&problem);
		simulated_annealing.Reset();
		timer.Restart();
		simulated_annealing.Search(10000);
		anneal_duration += timer.Stop();
		anneal_cost += simulated_annealing.best_cost();
		// std::cout << "simulated annealing: " << std::endl;
		// simulated_annealing.best_routes().Print();
		// std::cout << "total cost: " << simulated_annealing.best_cost() << std::endl;
		// std::cout << "computing time: " << timer.duration() << "ms" << std::endl;	
		// std::cout << std::endl;
	}
	std::cout << insert_cost / 100.0 << " " << insert_duration / 100.0 << std::endl;
	std::cout << local_cost / 100.0 << " " << local_duration / 100.0 << std::endl;
	std::cout << tabu_cost / 100.0 << " " << tabu_duration / 100.0 << std::endl;
	std::cout << anneal_cost / 100.0 << " " << anneal_duration / 100.0 << std::endl;
	
	return 0;
}