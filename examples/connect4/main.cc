#include "mcts.h"
#include "alpha_beta_search.h"
#include "rlop/common/timer.h"

int main(int argc, char *argv[]) {
    using namespace connect4;
    
    rlop::Timer timer;
   
    Board board;
    board.Reset();
   
    if (argc <= 1 || std::string(argv[1]) == "alpha_beta") {
        AlphaBetaSearch solver;
        std::ifstream input(argv[2]);
        std::string position;
        while(getline(input, position)) {                                            
            std::cout << "Solving position:" << std::endl;
            board.Reset(position);
            board.Print();
            solver.Reset();
            timer.Restart();
            board.MakeMove(solver.NewSearch(board));
            timer.Stop();
            board.Print();
            std::cout << "Solved in duration: " << timer.duration() << std::endl;
        }
    }
    else if (std::string(argv[1]) == "mcts") {
        board.Print();
        MCTS solver;
        solver.Reset();
        std::string input;
        while (input != "O" && input != "X") {
            std::cout << "choose player (O/X): ";
            std::cin >> input;
            std::cout << std::endl;
        }
        Int player = input == "X"? 0 : 1;
        board.Print();
        Int move;
        while(!board.IsFull() && !board.Win()) {
            if (board.num_moves() % 2 == player) {
                timer.Restart();
                board.MakeMove(solver.NewSearch(board));
                timer.Stop();
                std::cout << "duration: " << timer.duration() << std::endl;
            }
            else {
                if (board.IsFull() || board.Win()) 
                    break;
                std::cout << std::endl;
                std::cout << "next move: ";
                while(!(std::cin >> move) || !board.MakeMove(move)) {
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "ilegal move" << std::endl;
                    std::cout << std::endl;
                    std::cout << "next move: ";
                }
            }
            std::cout << std::endl;
            board.Print();
        }
        std::cout << std::endl;
        if (board.Win()) {
            if (board.num_moves() % 2 == 0) 
                std::cout << "winner: X" << std::endl;
            else
                std::cout << "winner: O" << std::endl; 
        }
        else if (board.IsFull())
            std::cout << "drawn" << std::endl;

    }
    return 0;
}