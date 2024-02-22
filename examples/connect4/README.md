# Connect4

Connect4 is a two-player connection board game, in which the players choose a color and then take turns dropping colored discs from the top into a seven-column, six-row vertically suspended grid. The pieces fall straight down, occupying the lowest available space within the column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs. For more details, please refer to https://en.wikipedia.org/wiki/Connect_Four.

## Run

1. **Compile**

    ```
    cd path/to/project
    mkdir build
    cd build
    cmake .. -DBUILD_CONNECT4=ON
    make
    ```
2. **Run**
   
    Solve game positions in positions.txt by alpha-beta search. The positions are arranged in order of solving time. The last position, starting from the initial position, takes about 10 hours to solve.

    ```
    ./examples/connect4/connect4 alpha_beta ../examples/connect4/positions.txt
    ```

    Play with MCTS agent.
    ```
    ./examples/connect4/connect4 mcts
    ```

