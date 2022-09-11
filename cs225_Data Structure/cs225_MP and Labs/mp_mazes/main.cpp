#include <iostream>
#include "dsets.h"
#include "maze.h"
#include "cs225/PNG.h"

using namespace std;

int main()
{
    // Write your own main here
    // cout << "Add your own tests here! Modify main.cpp" << endl;

    SquareMaze maze = SquareMaze();
    maze.makeMaze(3, 3);
    PNG *maze_img = maze.drawMaze();
    

    /*
    for (int i = 0; i < 3; i ++)
    {
        for (int j = 0; j < 3; j ++)
        {
            cout << "Point(" << i << ", " << j << "):" << endl;
            cout << "Go Right: " << maze.canTravel(i, j, 0) << endl;
            cout << "Go Down: " << maze.canTravel(i, j, 1) << endl;
            cout << "Go left: " << maze.canTravel(i, j, 2) << endl;
            cout << "Go Up: " << maze.canTravel(i, j, 3) << endl;
            cout << "\n";
        }
    }
    */

    
    vector<int> solution = maze.solveMaze();
    cout << "Solution: " << endl;
    for (unsigned int i = 0; i < solution.size(); i ++)
    {
        cout << solution[i] << " ";
    }
    cout << "\n";

    maze_img = maze.drawMazeWithSolution();
    maze_img -> writeToFile("test_maze.png");

    return 0;
}
