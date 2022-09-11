/* Your code here! */
#pragma once

#include <vector>
using namespace std;

class DisjointSets
{
    private:
        vector<int> vec;
    
    public:
        DisjointSets();
        ~DisjointSets();
        void clear_sets();
        void addelements(int num);
        int find(int elem);
        void setunion(int a, int b);
        int size(int elem);
        bool check_isolated(int size);
        void printvec() const;
};