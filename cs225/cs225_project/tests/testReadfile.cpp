#include "../cs225/catch/catch.hpp"

#include <cmath>
#include <vector>
#include <iostream>

#include "../Airport.h"
#include "../Readfile.h"
#include "../Route.h"

using namespace std;

/** 
    First test case with flawless input.
    Test reading every entry correctly.
**/
TEST_CASE("Test readfile_1", "[weight=10][part1]")
{
    vector<Airport> airports;
    vector<Route*> routes;
    vector<Route> route;

    Readfile read = Readfile();
    read.readfile_airport(airports, "./tests/airport_test.txt");
    read.readfile_routes(routes, airports, "./tests/routes_test.txt");

    for (unsigned int i = 0; i < routes.size(); i ++)
    {
        route.push_back(*(routes[i]));
    }

    REQUIRE("\"Kugaaruk Airport\"" == airports[0].getName());
    REQUIRE("\"Baker Lake Airport\"" == airports[3].getName());
	REQUIRE("\"Canada\"" == airports[0].getCountry());
    REQUIRE("\"Baie Comeau\"" == airports[1].getCity());
    REQUIRE(0 == airports[0].getID());
    REQUIRE(3 == airports[3].getID());
    REQUIRE(airports[0].getLatitude() == 68.534401);
    REQUIRE(airports[2].getLongitude() == -70.99639892578125);
    REQUIRE("\"YBC\"" == airports[1].getIATA());
    REQUIRE("\"CYBG\"" == airports[2].getICAO());


    REQUIRE(0 == route[0].getStartID());
    REQUIRE(2 == route[1].getStartID());
    REQUIRE(0 == route[2].getStartID());
    REQUIRE(3 == route[3].getStartID());
    REQUIRE(1 == route[0].getEndID());
    REQUIRE(3 == route[1].getEndID());
    REQUIRE(2 == route[2].getEndID());
    REQUIRE(1 == route[3].getEndID());
}


/** 
    Second test with missing entries (Airport name and city name). 
    Only the second and fourth airport are valid (Originally).
    Only the last route is valid (Originally).

    The ultimate vector storing aiports object should have size 2
    The route vector should only store 1 valid route
**/
TEST_CASE("Test readfile_2", "[weight=10][part1]")
{
    vector<Airport> airports;
    vector<Route*> routes;
    vector<Route> route;

    Readfile read = Readfile();
    read.readfile_airport(airports, "./tests/airport_test_1.txt");
    read.readfile_routes(routes, airports, "./tests/routes_test_1.txt");

    for (unsigned int i = 0; i < routes.size(); i ++)
    {
        route.push_back(*(routes[i]));
    }

    REQUIRE(airports.size() == 2);
    REQUIRE("\"Baie Comeau Airport\"" == airports[0].getName());
    REQUIRE("\"Baker Lake Airport\"" == airports[1].getName());
	REQUIRE("\"Canada\"" == airports[0].getCountry());
    REQUIRE("\"Baker Lake\"" == airports[1].getCity());
    REQUIRE(0 == airports[0].getID());
    REQUIRE(1 == airports[1].getID());
    REQUIRE(airports[0].getLatitude() == 49.13249969482422);
    REQUIRE(airports[1].getLongitude() == -96.077796936);
    REQUIRE("\"YBC\"" == airports[0].getIATA());
    REQUIRE("\"CYBK\"" == airports[1].getICAO());

    REQUIRE(route.size() == 1);
    REQUIRE(1 == route[0].getStartID());
    REQUIRE(0 == route[0].getEndID());
}

/** 
    Third test with missing entries of route data (Airport ID or IATA, ICAO). 
    Also test missing longitude and latitude for aiports.
    Only the first, third and fourth airport are valid (Originally).
    Only the second and third route are valid (Originally).

    The ultimate vector storing aiports object should have size 3
    The route vector should store 2 valid routes
**/
TEST_CASE("Test readfile_3", "[weight=10][part1]")
{
    vector<Airport> airports;
    vector<Route*> routes;
    vector<Route> route;

    Readfile read = Readfile();
    read.readfile_airport(airports, "./tests/airport_test_2.txt");
    read.readfile_routes(routes, airports, "./tests/routes_test_2.txt");

    for (unsigned int i = 0; i < routes.size(); i ++)
    {
        route.push_back(*(routes[i]));
    }

    REQUIRE(airports.size() == 3);
    REQUIRE("\"Kugaaruk Airport\"" == airports[0].getName());
    REQUIRE("\"CFB Bagotville\"" == airports[1].getName());
	REQUIRE("\"Canada\"" == airports[0].getCountry());
    REQUIRE("\"Bagotville\"" == airports[1].getCity());
    REQUIRE(0 == airports[0].getID());
    REQUIRE(1 == airports[1].getID());
    REQUIRE(2 == airports[2].getID());
    REQUIRE(airports[0].getLatitude() == 68.534401);
    REQUIRE(airports[1].getLongitude() == -70.99639892578125);
    REQUIRE("\"YBB\"" == airports[0].getIATA());
    REQUIRE("\"CYBK\"" == airports[2].getICAO());

    REQUIRE(route.size() == 2);
    REQUIRE(1 == route[0].getStartID());
    REQUIRE(0 == route[1].getStartID());
    REQUIRE(2 == route[0].getEndID());
    REQUIRE(1 == route[1].getEndID());
}

