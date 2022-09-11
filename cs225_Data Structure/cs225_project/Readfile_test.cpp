#include "Readfile.h"
#include "Graph.h"
#include "Airport.h"
// #include "Route.h"

// #include <bits/stdc++.h>
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
    vector<Route*> routes;
    vector<Route> route;

    vector<int> airports_id;
    string airport_name_1 = "Baie Comeau Airport";
    string airport_name_2 = "CFB Bagotville";
    string airport_1 = "\"" + airport_name_1 + "\"";
    string airport_2 = "\"" + airport_name_2 + "\"";

    Readfile read = Readfile();
    airports_id = read.readfile_airport(airports, "./data/airport.txt", airport_1, airport_2);
    read.readfile_routes(routes, airports, "./data/route.txt");

    for (unsigned int i = 0; i < routes.size(); i ++)
    {
        route.push_back(*(routes[i]));
    }

    cout << airports_id[0] << endl;
    cout << airports[airports_id[0]].getName() << endl;
    cout << airports[airports_id[0]].getCity() << endl;
    cout << airports_id[1] << endl;
    cout << airports[airports_id[1]].getName() << endl;
    cout << airports[airports_id[1]].getCity() << endl;
    
    // for (unsigned int i = 0; i < airports.size(); i ++)
    // {
    //     cout << airports[i].getID() << " ";
    //     cout << airports[i].getName() << " ";
    //     cout << airports[i].getCity() << " ";
    //     cout << airports[i].getIATA() << " ";
    //     cout << airports[i].getICAO() << " ";
    //     cout << airports[i].getCountry() << " ";
    //     cout << airports[i].getLatitude() << " ";
    //     cout << airports[i].getLongitude() << endl;
    // }
    // cout << "\n";

    
    // for (unsigned int j = 0; j < route.size(); j ++)
    // {
    //     cout << route[j].getStartID() << " ";
    //     cout << route[j].getEndID() << " ";
    //     cout << route[j].getDist() << endl;
    // }
    
    
    return 0;
}