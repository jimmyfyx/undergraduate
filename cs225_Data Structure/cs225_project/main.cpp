// main.cpp
#include "Readfile.h"
#include "Graph.h"
#include "Airport.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "cs225/PNG.h"
using namespace std;

int main()
{
    /* Initialize the airport vector */
    vector<Airport> airports;
    /* Initialize the routes vector */
    vector<Route> routes;
    vector<Route*> routes_ptrs;
    /* Initialize Readfile */
    vector<int> airports_id;
    Readfile read_file = Readfile();

    /** Read user input **/
    bool valid_airports = false;
    while (valid_airports == false)
    {
        string airport_name_1;
        string airport_name_2;
        cout << "Please enter your source airport name: ";
        getline(cin, airport_name_1);
        cout << "Please enter your destination airport name: ";
        getline(cin, airport_name_2);
        cout << "\n";
        string airport_1 = "\"" + airport_name_1 + "\"";
        string airport_2 = "\"" + airport_name_2 + "\"";

        /* Read from dataset */
        airports_id = read_file.readfile_airport(airports, "./data/airport.txt", airport_1, airport_2);
        read_file.readfile_routes(routes_ptrs, airports, "./data/route.txt");
        
        for (auto & ptr : routes_ptrs)
        {
            routes.push_back(*ptr);
        }

        /* Check whether the user input is valid */
        if (airports_id[0] == -1 && airports_id[1] != -1)
        {
            cout << "The system cannot find the source airport. Please enter a valid source airport name." << endl;
            cout << "\n";
        }
        else if (airports_id[0] != -1 && airports_id[1] == -1)
        {
            cout << "The system cannot find the destination airport. Please enter a valid destination airport name." << endl;
            cout << "\n";
        }
        else if (airports_id[0] == -1 && airports_id[1] == -1)
        {
            cout << "It seems that you plan to not travel on Earth. Please enter valid airport names." << endl;
            cout << "\n";
        }
        else
        {
            valid_airports = true;
        }
    }
    
    
    /* Construct the airport graph */
    cs225::PNG temp;
    temp.readFromFile("original.png");
    Graph AirportGraph(routes,airports,temp);

    /* Find shortest path */
    vector<unsigned> shortest_path;
    unsigned int source = (unsigned int)airports_id[0];
    unsigned int destination = (unsigned int)airports_id[1];
    shortest_path = AirportGraph.Dijkstra(source, destination);

    if (shortest_path[0] == 999999)
    {
        /* Cannont find a shortest path */
        cout << "Sorry, your travel plan is not possible. We cannot find a valid route between the two airports." << endl;
        return 0;
    }

    if (shortest_path.size() == 2)
    {
        cout << "---Stop: No stop" << endl;
    }
    else
    {
        cout << "---Stop: ";
        for (unsigned int i = 1; i < shortest_path.size() - 1; i ++)
        {
            cout << airports[shortest_path[i]].getName() << " ";
        }
        cout << "\n";
    }

    cout << "---Total Distance: ";
    cout << AirportGraph.total_dist << " km";
    cout << "\n"; 

    cout<<endl<<endl;
    string top;
    cout << "Please enter how many ranks you want, 10 if empty: ";
    getline(cin, top);
    cout<<endl;

    vector<size_t> rank = AirportGraph.PageRank(10,5);

    cout<<"Top"<<top<<" Popular Airports: "<<endl; 
    for (unsigned i = 0; i < rank.size() ; i++) {
        size_t index = rank[i];
        Airport ap(airports[index]);
        cout<<"No. "<<i+1<<" Airport is "<<ap.getName()<<" in "<<ap.getCity()<<", "<<ap.getCountry()<<endl;
    }
    cout<<endl;

    return 0;
}