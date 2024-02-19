#include "mcts.h"
#include "alpha_beta_search.h"
#include "rlop/common/timer.h"

int main() {
    using namespace connect4;

    Board board;
    board.Reset();
    // board.Reset("...............................X......O...");                                              
    // board.Reset(".................X......O......X......O...");                                              
    // board.Reset("..........O......X......O......X......OX..");                                              
    // board.Reset("..........O......X......OX.....XO.....OX..");                                              
    // board.Reset("..........OX.....XO.....OX.....XO.....OX..");                                        
    // board.Reset("..........OX.....XO.X...OX.O...XO.X...OX.O");                                              
    // board.Reset(".......X.OX...O.XXX.XX.XOO.OO.OXO.XOOXOXOO");
    board.Print();

    rlop::Timer timer;
    
    AlphaBetaSearch solver;
    solver.Reset();

    // MCTS solver;
    // solver.Reset();

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
    return 0;
}