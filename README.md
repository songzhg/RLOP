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

- **For reinforcement learning**: If your project involves algorithms that utilize reinforcement learning algorithms or deep learning models, the installation of libtorch is necessary.
  
  - **Installation of libtorch:**
    Follow the instructions on the official PyTorch website: https://pytorch.org/cppdocs/installing.html. Make sure to select the appropriate version for your operating system and CUDA version (if applicable) to ensure compatibility with your development environment.

- **For Gymnasium environments**: RLOP includes a C++ wrapper (GymEnv and GymVectorEnv class) for Gymnasium environments to support reinforcement learning algorithms in C++. To fully utilize this capability, specific requirements need to be met:
 
  - **Installation of Gymnasium:**
    Follow the official gymnasium documentation to set up the environments you need: https://github.com/Farama-Foundation/Gymnasium.

  - **Installation of pybind11:**
    Visit the pybind11 docoments for detailed installation instructions: https://pybind11.readthedocs.io/en/stable/installing.html. 
    
  - **Embedding Python interpreter by pybind11:** 
    Pybind11 is possible to embed the Python interpreter into a C++ program. Follow the instructions on the official: https://pybind11.readthedocs.io/en/stable/advanced/embedding.html.
  
## Run Examples

RLOP implements several benchmark problems and provides examples demonstrating how to solve these problems using the algorithms within RLOP. For the requirements and running method of each examples, please refer to the links in the table of [Examples](##Examples) below. 
  
  
## Implemented Algorithms

| **Algorithm**                         | **Type** |
| ---------------------------           | ----------------------|
| DQN                                   |   RL  |
| PPO                                   |   RL  |
| SAC                                   |   RL  |
| MCTS/PUCT                             |   Search |
| Root-parallel MCTS/PUCT               |   Search |
| Alpha-beta Search                     |   Search |
| Alpha-beta Search with Tranposition   |  Search |
| Local Search                          |  Opt |
| Tabu Search                           |  Opt |
| Simulated Annealing                   |  Opt |

## Examples

| **Problem**                                                               | **Algorithms/Methods** |
| ---------------------------                                               | ----------------------|
| [Lunar Lander](examples/lunar_lander/README.md)                           |   DQN, PPO  |
| [Continuous Lunar Lander](examples/continuous_lunar_lander/README.md)     |   PPO, SAC  |
| Snake                                                                     |   MCTS, DQN, PPO  |
| Connect4                                                                  |   Alpha-beta Search, MCTS |
| VRP                                                                       |   Insertion, Local Search, Tabu Search, Simulated Annealling |
| Multi-armed bandit                                                        |   UCB1, Epsilon-greedy, Softmax, ... |
