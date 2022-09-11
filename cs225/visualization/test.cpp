#include "Graph.h"
#include "Airport.h"
#include "Route.h"
#include "Readfile.h"
#include "cs225/PNG.h"
#include "cs225/HSLAPixel.h"
#include <vector>
#include <iostream>

//#include "../cs225/catch/catch.hpp"

int main()
{
    
    // Initialize the airport vector
    vector<Airport> airports;
    vector<int> airports_id;

    // Initialize the routes vector
    vector<Route> routes;
    vector<Route*> routes_ptrs;
    Readfile read = Readfile();
    string airport_name_1;
    string airport_name_2;
    cout << "Please enter your source airport name: ";
    getline(cin, airport_name_1);
    cout << "Please enter your destination airport name: ";
    getline(cin, airport_name_2);
    cout << "\n";
    string airport_1 = "\"" + airport_name_1 + "\"";
    string airport_2 = "\"" + airport_name_2 + "\"";
    airports_id = read.readfile_airport(airports, "./data/airport.txt", airport_1, airport_2);
    read.readfile_routes(routes_ptrs, airports, "./data/route.txt");

    for (auto & ptr : routes_ptrs)
    {
        routes.push_back(*ptr);
    }

    cs225::PNG temp;
    temp.readFromFile("original.png");
    Graph myGraph(routes,airports,temp);
    myGraph.BFS();
    std::vector<size_t> rk = myGraph.PageRank(10,4,0.85);
    unsigned int source = (unsigned int)airports_id[0];
    std::cout<<source<<std::endl;
    unsigned int destination = (unsigned int)airports_id[1];
    std::cout<<destination<<std::endl;
    std::vector<unsigned> path = myGraph.Dijkstra(source, destination);
    myGraph.draw_rank(rk);
    myGraph.drawLine(path);
    cs225::PNG output = myGraph.getBasePic();
    output.writeToFile("test_output.png");
}
