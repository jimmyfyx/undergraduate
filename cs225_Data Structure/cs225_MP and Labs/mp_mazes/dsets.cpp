/* Your code here! */
#include "dsets.h"
#include <vector>
#include <iostream>
using namespace std;


DisjointSets::DisjointSets()
{
    /* Nothing */
}


DisjointSets::~DisjointSets()
{
    /* Nothing */
}


/**
    Helper function to clear the existing DisjointSets object
**/
void DisjointSets::clear_sets()
{
    vec.clear();
}


void DisjointSets::addelements(int num)
{
    for (int i = 0; i < num; i ++)
    {
        // Push new roots to the vector
        vec.push_back(-1);
    }
}


int DisjointSets::find(int elem)
{
    if (vec[elem] < 0)
    {
        // The element is the root node
        return elem;
    }
    else
    {
        int root = find(vec[elem]);
        // Directly point the node 'elem' to its root
        vec[elem] = root;
        return root;
    }
}


void DisjointSets::setunion(int a, int b)
{
    // Find the roots of 'a' and 'b'
    int root_a = find(a);
    int root_b = find(b);

    if (vec[root_a] == vec[root_b])
    {
        int temp = vec[root_b];
        vec[root_b] = root_a;
        // Update the size of the tree
        vec[root_a] = vec[root_a] + temp;
    }
    else if (vec[root_a] < vec[root_b])
    {
        int temp = vec[root_b];
        vec[root_b] = root_a;
        // Update the size of the tree
        vec[root_a] = vec[root_a] + temp;
    }
    else
    {
        int temp = vec[root_a];
        vec[root_a] = root_b;
        // Update the size of the tree
        vec[root_b] = vec[root_b] + temp;
    }
}


int DisjointSets::size(int elem)
{
    int root = find(elem);
    return vec[root] * -1;
}


/**
    Helper function to check whether every cell is connected to the maze
    Return true if there are, otherwise return false
**/
bool DisjointSets::check_isolated(int size)
{
    bool isolated = true;
    if (vec.size() != 0)
    {
        for (int i = 0; i < size; i ++)
        {
            if (vec[i] == -1 * size)
            {
                isolated = false;
                break;
            }
        }
    }
    
    return isolated;
}


void DisjointSets::printvec() const
{
    for (unsigned int i = 0; i < vec.size(); i ++)
    {
        cout << vec[i] << " ";
    }
    cout << "\n";
}

