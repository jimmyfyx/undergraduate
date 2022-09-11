/* Your code here! */
#include "maze.h"
#include "dsets.h"
#include "cs225/PNG.h"

#include <queue>
#include <stack>
#include <stdlib.h> 
#include <time.h>
#include <iostream>
using namespace std;
using namespace cs225;


SquareMaze::SquareMaze()
{
    width_ = 0;
    height_ = 0;
    maze_ = DisjointSets();
}


SquareMaze::~SquareMaze()
{
    /* Nothing */
}


void SquareMaze::makeMaze(int width, int height)
{
    if (width_ != 0 || height_ != 0)
    {
        // Clear the existing maze
        width_ = 0;
        height_ = 0;
        maze_.clear_sets();
        right_neigh.clear();
        down_neigh.clear();
    }
    
    width_ = width;
    height_ = height;

    // Initialize the DisjointSets representing the maze
    maze_.addelements(width_ * height_);

    // Initialize the neighbors vector
    for (int i = 0; i < width_ * height_; i ++)
    {
        right_neigh.push_back(-1);
        down_neigh.push_back(-1);
    }

    while (true)
    {
        // Check whether there are isolated cells
        if (maze_.size(0) >= width_ * height_)
        {
            break;
        }
        else
        {
            // Randomly generate the x, y coordinates
            // srand(time(NULL));
            int x = rand() % width_ ;
            int y = rand() % height_;

            if (y != height_ - 1)
            {
                if (maze_.find(x + y * width_) != maze_.find(x + (y + 1) * width_))
                {
                    maze_.setunion(x + y * width_, x + (y + 1) * width_);
                    down_neigh[x + y * width_] = x + (y + 1) * width_;
                }
            }

            if (x != width_ - 1)
            {
                if (maze_.find(x + y * width_) != maze_.find((x + 1) + y * width_))
                {
                    maze_.setunion(x + y * width_, (x + 1) + y * width_);
                    right_neigh[x + y * width_] = (x + 1) + y * width_;
                }
            }
        }   
    }
}


PNG* SquareMaze::drawMaze() const
{
    int image_width = width_ * 10 + 1;
    int image_height = height_ * 10 + 1;
    PNG *maze_image = new PNG(image_width, image_height);

    // Blacken topmost row and leftmost column
    for (int i = 10; i < image_width; i ++)
    {
        HSLAPixel & pixel = maze_image -> getPixel(i, 0);
        pixel.l = 0;
    }

    for (int j = 0; j < image_height; j ++)
    {
        HSLAPixel & pixel = maze_image -> getPixel(0, j);
        pixel.l = 0;
    }

    // Draw the actual maze
    for (int i = 0; i < width_ * height_; i ++)
    {
        // For each maze cell
        // Calculate its maze coordinates (x, y)
        int x = i % width_;
        int y = i / width_;

        
        if (right_neigh[i] == -1)
        {
            // Exist right wall
            for (int k = 0; k <= 10; k ++)
            {
                HSLAPixel & pixel = maze_image -> getPixel((x + 1) * 10, y * 10 + k);
                pixel.l = 0;
            }
        }

        if (down_neigh[i] == -1)
        {
            // Exist bottom wall
            for (int k = 0; k <= 10; k ++)
            {
                HSLAPixel & pixel = maze_image -> getPixel(x * 10 + k, (y + 1) * 10);
                pixel.l = 0;
            }
        }

    }

    return maze_image;
}


bool SquareMaze::canTravel(int x, int y, int dir) const
{
    // Convert x, y coordinates to cell index
    int index = width_ * y + x;

    if (dir == 0)
    {
        // Check possible right
        if (right_neigh[index] == -1)
        {
            return false;
        }
        return true;
    }
    else if (dir == 1)
    {
        // Check possible down
        if (down_neigh[index] == -1)
        {
            return false;
        }
        return true;
    }
    else if (dir == 2)
    {
        // Check possible left
        if (index % width_ == 0)
        {
            return false;
        }
        else
        {
            if (right_neigh[index - 1] == -1)
            {
                return false;
            }
            return true;
        }
    }
    else if (dir == 3)
    {
        // Check possible up
        if (y == 0)
        {
            return false;
        }
        else
        {
            if (down_neigh[index - width_] == -1)
            {
                return false;
            }
            return true;
        }
    }

    return false;
}



void SquareMaze::setWall(int x, int y, int dir, bool exists)
{
    // Convert x, y coordinates to cell index
    int index = width_ * y + x;

    if (dir == 0)
    {
        if (exists == true)
        {
            // Set right wall
            right_neigh[index] = -1;
        }
        else
        {
            right_neigh[index] = index + 1;
        }
    }
    else
    {
        if (exists == true)
        {
            // Set bottom wall
            down_neigh[index] = -1;
        }
        else
        {
            down_neigh[index] = index + width_;
        }
    }
}



vector<int> SquareMaze::solveMaze()
{
    // Initialze the predecessor vector and visited vector
    for (int i = 0; i < width_ * height_; i ++)
    {
        predecessor.push_back(-1);
        visited.push_back(false);
    }
    
    queue<int> q;
    visited[0] = true;
    q.push(0);

    while (q.empty() == false)
    {
        // Get the current cell index
        int cur = q.front();
        q.pop();
        int x = cur % width_;
        int y = cur / width_;

        // Check whether the current cell is a cell in the bottom row
        if (cur >= width_ * height_ - width_ && cur <= width_ * height_ - 1)
        {
            // Find a solution
            // Record the solution
            stack<int> solution;
            vector<int> sol;
            int direction = 0;
            
            int pred = predecessor[cur];
            if (pred == cur - 1)
            {
                // Direction is right
                direction = 0;
            }
            else if (pred == cur + 1)
            {
                // Direction is left
                direction = 2;
            }
            else if (pred == cur - width_)
            {
                // Direction is down
                direction = 1;
            }
            else if (pred == cur + width_)
            {
                // Direction is up
                direction = 3;
            }
            solution.push(direction);


            while (true)
            {
                int cur_cell = pred;
                pred = predecessor[cur_cell];
                if (pred == -1)
                {
                    break;
                }
                else
                {
                    if (pred == cur_cell - 1)
                    {
                        // Direction is right
                        direction = 0;
                    }
                    else if (pred == cur_cell + 1)
                    {
                        // Direction is left
                        direction = 2;
                    }
                    else if (pred == cur_cell - width_)
                    {
                        // Direction is down
                        direction = 1;
                    }
                    else if (pred == cur_cell + width_)
                    {
                        // Direction is up
                        direction = 3;
                    }
                    solution.push(direction);
                }
            }

            // Build the current possible solution vector
            unsigned int size = solution.size();
            for (unsigned int i = 0; i < size; i ++)
            {
                sol.push_back(solution.top());
                solution.pop();
            }

            // Push the current solution to the possible solutions vector
            possi_sol.push_back(sol);
            last.push_back(x);
        }

        if (canTravel(x, y, 0) == true)
        {
            // The adjacent right cell
            if (visited[cur + 1] == false)
            {
                visited[cur + 1] = true;
                q.push(cur + 1);
                // Mark the current cell as its predecessor
                predecessor[cur + 1] = cur;
            }
        }

        if (canTravel(x, y, 1) == true)
        {
            // The adjacent bottom cell
            if (visited[cur + width_] == false)
            {
                visited[cur + width_] = true;
                q.push(cur + width_);
                // Mark the current cell as its predecessor
                predecessor[cur + width_] = cur;
            }
        }

        if (canTravel(x, y, 2) == true)
        {
            // The adjacent left cell
            if (visited[cur - 1] == false)
            {
                visited[cur - 1] = true;
                q.push(cur - 1);
                // Mark the current cell as its predecessor
                predecessor[cur - 1] = cur;
            }
        }

        if (canTravel(x, y, 3) == true)
        {
            // The adjacent up cell
            if (visited[cur - width_] == false)
            {
                visited[cur - width_] = true;
                q.push(cur - width_);
                // Mark the current cell as its predecessor
                predecessor[cur - width_] = cur;
            }
        }
    }

    cout << "Reach" << endl;

    // The traversal is done, choose the longest path
    unsigned int max_path_size = 0;
    unsigned int max_idx = 0;
    int x_cor = 100000; 
    for (unsigned int i = 0; i < possi_sol.size(); i ++)
    {
        if (possi_sol[i].size() > max_path_size)
        {
            max_path_size = possi_sol[i].size();
            max_idx = i;
            x_cor = last[i];
        }
        else if (possi_sol[i].size() == max_path_size)
        {
            if (last[i] < x_cor)
            {
                max_idx = i;
                max_path_size = possi_sol[i].size();
                x_cor = last[i];
            }
        }
    }

    return possi_sol[max_idx];
}



PNG* SquareMaze::drawMazeWithSolution()	
{
    PNG *maze = drawMaze();
    vector<int> solution = solveMaze();

    // Mark the start point as red
    int cur_x = 5;
    int cur_y = 5;
    int cur_index = 0;
    HSLAPixel & pixel = maze -> getPixel(cur_x, cur_y);
    pixel.h = 0;
    pixel.s = 1;
    pixel.l = 0.5;
    pixel.a = 1;

    // Draw the solution
    for (unsigned int i = 0; i < solution.size(); i ++)
    {
        if (solution[i] == 0)
        {
            // Go right
            for (int j = 1; j <= 10; j ++)
            {
                HSLAPixel & pixel = maze -> getPixel(cur_x + j, cur_y);
                pixel.h = 0;
                pixel.s = 1;
                pixel.l = 0.5;
                pixel.a = 1;
            }
            cur_x = cur_x + 10;
            cur_index = cur_index + 1;
        }
        else if (solution[i] == 1)
        {
            // Go down
            for (int j = 1; j <= 10; j ++)
            {
                HSLAPixel & pixel = maze -> getPixel(cur_x, cur_y + j);
                pixel.h = 0;
                pixel.s = 1;
                pixel.l = 0.5;
                pixel.a = 1;
            }
            cur_y = cur_y + 10;
            cur_index = cur_index + width_;
        }
        else if (solution[i] == 2)
        {
            // Go left
            for (int j = 1; j <= 10; j ++)
            {
                HSLAPixel & pixel = maze -> getPixel(cur_x - j, cur_y);
                pixel.h = 0;
                pixel.s = 1;
                pixel.l = 0.5;
                pixel.a = 1;
            }
            cur_x = cur_x - 10;
            cur_index = cur_index - 1;
        }
        else if (solution[i] == 3)
        {
            // Go up
            for (int j = 1; j <= 10; j ++)
            {
                HSLAPixel & pixel = maze -> getPixel(cur_x, cur_y - j);
                pixel.h = 0;
                pixel.s = 1;
                pixel.l = 0.5;
                pixel.a = 1;
            }
            cur_y = cur_y - 10;
            cur_index = cur_index - width_;
        }
    }

    // Generate the destination coordinates
    int x = cur_index % width_;
    int y = cur_index / width_;

    // Mark the exit
    for (int k = 1; k <= 9; k ++)
    {
        HSLAPixel & pixel = maze -> getPixel(x * 10 + k, (y + 1) * 10);
        pixel.l = 1;
    }

    return maze;
}


