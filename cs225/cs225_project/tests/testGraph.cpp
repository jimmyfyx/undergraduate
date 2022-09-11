/**
 * @file testGraph.cpp
 * @author Zisu Jiang
 * @brief Test constructor and simplifiedPageRank in graph.cpp 
 * @version 0.1
 * @date 2022-05-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../Graph.h"
#include "../cs225/catch/catch.hpp"
#include "../Airport.h"
#include "../Readfile.h"



TEST_CASE("CTR::testGraphConstructor","[part3]")
{
    vector<Route> routes;
    vector<Airport> airports;
    vector<Route*> routes_ptrs;

    Readfile read = Readfile();
    read.readfile_airport(airports, "./tests/airport_test_3.txt");
    read.readfile_routes(routes_ptrs, airports, "./tests/routes_test_3.txt");
    
    for (auto & ptr : routes_ptrs) {
        routes.push_back(*ptr);
    }

    Graph myGraph(routes,airports);
    myGraph.printGraph();
}

TEST_CASE("test_bfs_simple")
{
    vector<Route> routes;
    vector<Airport> airports;
    airports.push_back(Airport());
    for(int i = 0 ; i < 3 ; i++){
        Route route(0,0,0);
        route.setStartID(i);
        route.setEndID((i+1)%3);
        route.setDist(5 + i);
        routes.push_back(route);
        //airports.push_back(Airport());
        string name = "Tdfnri";
        Airport a(name, "U.S.", "Champaign", i, 100, 200);
        airports.push_back(a);
    }
    Graph g(routes, airports);
    g.BFS();
}

TEST_CASE("PR::testPageRank","[part3]") {
    vector<Route> routes;
    vector<Airport> airports;
    vector<Route*> routes_ptrs;
    vector<size_t> rank;

    Readfile read = Readfile();
    read.readfile_airport(airports, "./data/airport.txt");
    read.readfile_routes(routes_ptrs, airports, "./data/route.txt");
    
    for (auto & ptr : routes_ptrs) {
        routes.push_back(*ptr);
    }

    Graph myGraph(routes,airports);

    int top = 10;
    rank = myGraph.PageRank(top,5);

    cout<<endl<<endl;
 
    cout<<"Top "<<top<<" Popular Airports: "<<endl; 
    for (unsigned i = 0; i < rank.size() ; i++) {
        cout<<"    ";
        size_t index = rank[i];
        Airport ap(airports[index]);
        cout<<"No. "<<i+1<<" Airport is "<<ap.getName()<<" in "<<ap.getCity()<<", "<<ap.getCountry()<<endl;
    }
    cout<<endl;
    rank.clear();

    rank = myGraph.simplifiedPageRank();
    cout<<endl<<endl;
 
    cout<<"Top "<<top<<" Popular Airports: "<<endl; 
    for (unsigned i = 0; i < rank.size() ; i++) {
        cout<<"    ";
        size_t index = rank[i];
        Airport ap(airports[index]);
        cout<<"No. "<<i+1<<" Airport is "<<ap.getName()<<" in "<<ap.getCity()<<", "<<ap.getCountry()<<endl;
    }
    cout<<endl;

}