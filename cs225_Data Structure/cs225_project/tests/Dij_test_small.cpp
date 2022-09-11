// main.cpp
#include "Readfile.h"
#include "Graph.h"
#include "Airport.h"
// #include "Route.h"

#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

int main()
{
    // Initialize the airport vector
    vector<Airport> airports;

    // Initialize the routes vector
    vector<Route> routes;
    vector<Route*> routes_ptrs;
    //string filename = "./tests/airport_test.txt";
    Readfile read = Readfile();
    read.readfile_airport(airports, "./tests/Dij_Airport_test1.txt");
    read.readfile_routes(routes_ptrs, airports, "./tests/Dij_Route_test1.txt");
    
    for (auto & route : routes_ptrs)
    {
        routes.push_back(*route);
    }

    Graph myGraph(routes,airports);
    myGraph.printGraph();

    //Test Case 1: No path
    cout<<" "<<endl;
    cout<<"===================================================="<<endl;
    cout<<"Test Case 1: from airport YBC(1) to YBB(0), no path between"<<endl;
    cout<<"Expected Output: total_distance = 0"<<endl;
    cout<<"Expected Output: path = [999999]"<<endl;

    vector<unsigned> path = myGraph.Dijkstra(1, 0);
    cout<<"total_distance:"<<myGraph.total_dist<<endl;

    cout<<"path:"<<endl;
    for (int i = 0; i < path.size(); i++){
       cout<<path[i]<<"  ";
    }
    cout<<" "<<endl;

    //Test Case 2: Has path
    cout<<" "<<endl;
    cout<<"===================================================="<<endl;
    cout<<"Test Case 2: from airport YBB(0) to YBK(3), has a path between"<<endl;
    cout<<"Expected Output: total_distance = 546.762"<<endl;
    cout<<"Expected Output: path = [0  3]"<<endl;

    path = myGraph.Dijkstra(0, 3);
    cout<<"total_distance:"<<myGraph.total_dist<<endl;

    cout<<"path:"<<endl;
    for (int i = 0; i < path.size(); i++){
       cout<<path[i]<<"  ";
    }
    cout<<" "<<endl;

    //Test Case 3: Test directed path
    cout<<" "<<endl;
    cout<<"===================================================="<<endl;
    cout<<"Test Case 3: from airport YBK(3) to YBB(0), has a path from YBB to YBK, but no path from YBK to YBB"<<endl;
    cout<<"Expected Output: total_distance = 0"<<endl;
    cout<<"Expected Output: path = [999999]"<<endl;

    path = myGraph.Dijkstra(3, 0);
    cout<<"total_distance:"<<myGraph.total_dist<<endl;

    cout<<"path:"<<endl;
    for (int i = 0; i < path.size(); i++){
       cout<<path[i]<<"  ";
    }
    cout<<" "<<endl;

    //Test Case 4: Test shortest path
    cout<<" "<<endl;
    cout<<"===================================================="<<endl;
    cout<<"Test Case 4: from airport YBC(1) to YAY(6), has 2 path from YBC to YAY, the path passed through 2 mid airports is shorter than the path passed through 1 mid airport"<<endl;
    cout<<"Expected Output: total_distance = 1859.08"<<endl;
    cout<<"Expected Output: path = [1  2  4  5]"<<endl;

    path = myGraph.Dijkstra(1, 5);
    cout<<"total_distance:"<<myGraph.total_dist<<endl;

    cout<<"path:"<<endl;
    for (int i = 0; i < path.size(); i++){
       cout<<path[i]<<"  ";
    }
    cout<<" "<<endl;

    return 0;
}
