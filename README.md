# RLOP: A Framework for Reinforcement Learning, Optimization and Planning Algorithms

RLOP is a generic and lightweight framework for reinforcement learning (RL), optimization (Opt), and planning/search algorithms in C++, aimed at simplifying studying, comparing and integrating algorithms across domains. 

In artificial intelligence, reinforcement learning, optimization, and planning/search have relatively independent research background and application scope, but also have similarities and overlaps. They complement each other in solving complex decision-making problems, offering opportunities for cross-disciplinary integration.

RLOP implements state-of-art algorithms in reinforcement learning, optimization, and planning/search in a domain-independent manner, in order to enable flexible customization and efficient integration across different domains.

## Main Features
- **Simple**: RLOP implements only the core logic of the algorithm, with other improvements to the algorithm being added in an extensible manner.
  
- **Domain-independent**: Algorithms are fully encapsulated within a class, abstracting away domain-specific details by defining interface functions, rather than directly accessing information from problem-specific classes.

## Installation

RLOP is a Header-Only C++ framework, so it does not require compiling or linking against precompiled binaries. To use this library in your project, follow these steps:

1. Clone the repository:

    ```
    git clone https://github.com/songzhg/RLOP.git
    ```

2. Include the header files:
    ```
    #include "rlop/path/to/algorithm.h"
    ```

## Requirements

This framework is built on C++17 and above. Ensure you have a C++ compiler that supports C++17 standards or higher. So far, it is only tested on Linux platforms.

- **For reinforcement learning**: If your project involves algorithms that utilize reinforcement learning or deep learning models, the installation of libtorch is necessary.
  
  - **Installation of libtorch:**
    Follow the instructions on the official PyTorch website: https://pytorch.org/cppdocs/installing.html. Make sure to select the appropriate version for your operating system and CUDA version (if applicable).

- **For Gymnasium environments**: RLOP includes a C++ wrapper (GymEnv and GymVectorEnv class) for Gymnasium environments to support reinforcement learning algorithms in C++. To fully utilize this capability, specific requirements need to be met:
 
  - **Installation of Gymnasium:**
    Follow the official gymnasium documentation to set up the environments you need: https://github.com/Farama-Foundation/Gymnasium.

  - **Installation of pybind11:**
    Visit the pybind11 docoments for detailed installation instructions: https://pybind11.readthedocs.io/en/stable/installing.html. 
    
  - **Embedding Python interpreter by pybind11:** 
    Pybind11 is possible to embed the Python interpreter into a C++ program. Follow the instructions on the official: https://pybind11.readthedocs.io/en/stable/advanced/embedding.html.
  
## Run Examples

RLOP implements several benchmark problems and provides examples demonstrating how to solve these problems using the algorithms within RLOP. For the requirements and running method of each examples, please refer to the links in the table of [Examples](#examples) below. 
  
  
## Implemented Algorithms

| **Algorithm**                         | **Type** |  **Reference** |
| ---------------------------           | ---------|  -------------|
| DQN                                   |   RL     |  [Mnih et al. 2015](https://www.nature.com/articles/nature14236) |
| PPO                                   |   RL     |  [Schulman et al. 2018](https://arxiv.org/abs/1707.06347)        |
| SAC                                   |   RL     |  [Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)        |
| MCTS/PUCT                             |   Search |  [Coulom 2006](https://hal.inria.fr/inria-00116992/document), [UCT paper](http://ggp.stanford.edu/readings/uct.pdf) |
| Root-parallel MCTS/PUCT               |   Search |
| Alpha-beta Search                     |   Search |  [Wikipedia1](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning), [Wikipedia2](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) |
| Alpha-beta Search with Tranposition   |  Search  |  [ChessProgramming](https://www.chessprogramming.org/Transposition_Table)  |
| Local Search                          |  Opt     |  [Wikipedia](https://en.wikipedia.org/wiki/Local_search_(optimization))  |
| Tabu Search                           |  Opt     |  [Wikipedia](https://en.wikipedia.org/wiki/Tabu_search#:~:text=Tabu%20search%20(TS)%20is%20a,1986%20and%20formalized%20in%201989.), Glover. 1986  |
| Simulated Annealing                   |  Opt     |  [Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing), Scott et al. 1983 |

## Examples

| **Problem**                                                               | **Algorithms/Methods** |
| ---------------------------                                               | ----------------------|
| [Lunar Lander](examples/lunar_lander/README.md)                           |   DQN, PPO  |
| [Continuous Lunar Lander](examples/continuous_lunar_lander/README.md)     |   PPO, SAC  |
| [Snake](examples/snake/README.md)                                         |   MCTS, DQN, PPO  |
| [Connect4](examples/connect4/README.md)                                   |   Alpha-beta Search, MCTS |
| [VRP](examples/vrp/README.md)                                             |   Insertion, Local Search, Tabu Search, Simulated Annealling |
| [Multi-armed bandit](examples/multi_armed_bandit/README.md)               |   UCB1, Epsilon-greedy, Softmax, ... |
