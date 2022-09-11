/* Your code here! */
#pragma once

#include "cs225/PNG.h"
#include "dsets.h"
#include <vector>
using namespace std;
using namespace cs225;

class SquareMaze
{
    private:
        int width_;
        int height_;
        DisjointSets maze_;
        vector<int> right_neigh;
        vector<int> down_neigh;
        vector<bool> visited;
        vector<int> predecessor;
        vector<vector<int>> possi_sol;
        vector<int> last;

    public:
        SquareMaze();
        ~SquareMaze();
        void makeMaze(int width, int height);
        bool canTravel(int x, int y, int dir) const;
        void setWall(int x, int y, int dir, bool exists);
        vector<int> solveMaze();
        PNG* drawMaze() const;
        PNG* drawMazeWithSolution();
};