# VRP

VRP (Vehicle Routing Problem) is a classic problem in the field of operations research and logistics. The VRP seeks to find the optimal routes for a fleet of vehicles to deliver goods or services to a set of customers, starting and finishing at a depot, under certain constraints. The goal is to minimize the total distance traveled, the total cost, or the number of vehicles used, while ensuring that each customer's demand is met. For more details, please refer to: https://en.wikipedia.org/wiki/Vehicle_routing_problem.

## Run

1. **Compile**

    ```
    cd path/to/project
    mkdir build
    cd build
    cmake .. -DBUILD_VRP=ON
    make
    ```
2. **Run**
   
    Run insertion, local search, tabu search and simulated annealing to solve a randomly generated VRP instance, where each vehicle has its own depot.
    ```
    ./examples/vrp/vrp
    ```
    

