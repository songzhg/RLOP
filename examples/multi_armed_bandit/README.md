# Multi-armed Bandit

A multi-armed bandit (MAB) problem is a classic scenario in probability theory and reinforcement learning that models the dilemma of balancing exploration and exploitation. The name "multi-armed bandit" comes from imagining a gambler at a row of slot machines (sometimes referred to as "one-armed bandits"), where each machine provides a different, unknown reward distribution. The gambler seeks to maximize their winnings by deciding which machines to play, how many times to play each machine, and in what order. For more details, please refer to: https://en.wikipedia.org/wiki/Multi-armed_bandit.

## Run

1. **Compile**

    ```
    cd path/to/project
    mkdir build
    cd build
    cmake .. -BUILD_MULTI_ARMED_BANDIT=ON
    make
    ```
2. **Run**
   
    Run epsilon greed, softmax, UCB1, persuit, and persuit epsilon greedy methods. This example takes 2000 times experiments on 1000 steps' bandit for each method. The average reward and the percentage of optimal actions taken by each step are recorded into result files (reward_results.txt and opt_results.txt). You can use the python funtion "csv_to_line_chart" provided in python/utils.py to plot these results.

    ```
    ./examples/multi_armed_bandit/multi_armed_bandit
    ```

    

