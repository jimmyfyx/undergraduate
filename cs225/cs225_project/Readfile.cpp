/**
    This file contains all functions that are in the Readfile class and will be used to read databases 
    files (airport and routes information). The process for checking potential data error is also implemented. 
    The functions will store complete data into pre-implemented data structures. 

    The corresponding header file is 'Readfile.h'
**/
#include "Readfile.h"

#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
using namespace std;


Readfile::Readfile()
{
    /* Nothing */
}


Readfile::~Readfile()
{
    /* Nothing */
}


/** 
    Helper function converting degrees to radians.

    @param degree Input degree value
    @return The radian value of the corresponding degree value
**/
long double Readfile::toRadians(const long double degree)
{
    long double one_deg = (M_PI) / 180;
    return (one_deg * degree);
}


/**
    The helper function calculating distance between two points on Earth.

    @param lat1 The latitude of the first point
    @param long1 The longitude of the first point
    @param lat2 The latitude of the second point
    @param long2 The longitude of the second point
    @return The acutal distance between the two points in km.
**/
long double Readfile::distance(long double lat1, long double long1, long double lat2, long double long2)
{
    /* Convert the latitudes and longitudes from degree to radians. */
    lat1 = toRadians(lat1);
    long1 = toRadians(long1);
    lat2 = toRadians(lat2);
    long2 = toRadians(long2);
     
    // Haversine Formula
    long double dlong = long2 - long1;
    long double dlat = lat2 - lat1;
 
    long double ans = pow(sin(dlat / 2), 2) + cos(lat1) * cos(lat2) * pow(sin(dlong / 2), 2);
 
    ans = 2 * asin(sqrt(ans));
 
    /** 
        Radius of Earth in Kilometers, R = 6371
        Use R = 3956 for miles
    **/
    long double R = 6371;
     
    // Calculate the result
    ans = ans * R;
 
    return ans;
}

/** 
    Helper function to check whether a string is a number. 

    @param string The string that needs to be checked
    @return true if the string is a number, otherwise false
**/
bool Readfile::isNumber(const string& str)
{
    for (char const &c : str) 
    {
        if (std::isdigit(c) == 0) return false;
    }
    return true;
}



/**
    The readfile_airport function accepts a vector storing Airport objects, and 
    it reads every line of the input file (airport database). When reading every line, it will check the validity
    of the data and construct a corresponding Airport object. If every entry is valid, the newly
    constructed Airport object will be pushed into 'airport_vec', otherwise not. 

    @param airport_vec The vector that stores every Airport object
    @param filepath The path and name of the input file
    @param airport_1 The first airport name of user input
    @param airport_2 The second airport name of user input
    @return A vector contains two integers, which are the ids of the two airports repectively
**/
vector<int> Readfile::readfile_airport(vector<Airport> & airport_vec, string filepath, string airport_1, string airport_2)
{
    airport_1_id = -1;
    airport_2_id = -1;
    // Open a new file to perform read operation line by line
    fstream newfile;
    newfile.open(filepath, ios::in);
    if (newfile.is_open() == true)
    {
        string line;
        while(getline(newfile, line))
        {
            // Parse each line and check for data error
            parse_correct_airport(line, airport_vec, airport_1, airport_2);
        }
        newfile.close(); 
    }

    /* Return user input airports id */
    vector<int> airports_id;
    airports_id.push_back(airport_1_id);
    airports_id.push_back(airport_2_id);
    return airports_id;
}


/**
    This is the overloading function of readfile_airport for test cases
**/
void Readfile::readfile_airport(vector<Airport> & airport_vec, string filepath)
{
    // Open a new file to perform read operation line by line
    fstream newfile;
    newfile.open(filepath, ios::in);
    if (newfile.is_open() == true)
    {
        string line;
        while(getline(newfile, line))
        {
            // Parse each line and check for data error
            parse_correct_airport(line, airport_vec);
        }
        newfile.close(); 
    }
}


/**
    The readfile_routes function accepts a vector storing Route struct, and 
    it reads every line of the input file (route database). When reading every line, it will check the validity
    of the data and instantiate a corresponding Route struct. If every entry is valid, the newly
    constructed Route struct will be pushed into 'routes_vec', otherwise not. 

    @param routes_vec The vector that stores every Route struct
    @param filepath The path and name of the input file
**/
void Readfile::readfile_routes(vector<Route*> & routes_vec, vector<Airport> & airport_vec, string filepath)
{
    // Open a new file to perform read operation line by line
    fstream newfile;
    newfile.open(filepath, ios::in);
    if (newfile.is_open() == true)
    {
        string line;
        while(getline(newfile, line))
        {
            // Parse each line and check for data error
            parse_correct_routes(line, routes_vec, airport_vec);
        }
        newfile.close(); 
    }
}


/**
    parse_correct_airport is the helper function that parses strings from every line of the file, and
    checks for data validity. 

    @param airport_vec The vector that stores every Airport object
    @param airport_1 The first airport name of user input
    @param airport_2 The second airport name of user input
    @param s Every line of the input file
**/
void Readfile::parse_correct_airport(string s, vector<Airport> & airport_vec, string airport_1, string airport_2)
{
    // Construct an airport object
    Airport *airport = new Airport();

    stringstream s_stream(s);
    int count = 0;
    bool airport_valid = true;
    unsigned int orig_id = 0;
    while(s_stream.good())
    {
        // 'substr' is the string of every parse (different portions of the line)
        string substr;
        getline(s_stream, substr, ',');
        
        if (count == 0)
        {
            // Read the airport's original id
            orig_id = std::stoul(substr);
        }

        if (count == 1)
        {
            // Check for valid airport name
            if (substr == "")
            {
                airport_valid = false;
                break;
            }

            if (substr[0] != '\"' || substr[substr.length() - 1] != '\"')
            {
                airport_valid = false;
                break;
            }

            airport -> setName(substr);
        }

        if (count == 2)
        {
            // Check for valid city name
            if (substr == "")
            {
                airport_valid = false;
                break;
            }

            if (substr[0] != '\"' || substr[substr.length() - 1] != '\"')
            {
                airport_valid = false;
                break;
            }

            airport -> setCity(substr);
        }

        if (count == 3)
        {
            // Check for valid country name
            if (substr == "")
            {
                airport_valid = false;
                break;
            }

            if (substr[0] != '\"' || substr[substr.length() - 1] != '\"')
            {
                airport_valid = false;
                break;
            }

            airport -> setCountry(substr);
        }

        if (count == 4)
        {
            // Add IATA (IATA can be \N if not assigned)
            airport -> setIATA(substr);
        }

        if (count == 5)
        {
            // Add ICAO (ICAO can be \N if not assigned)
            airport -> setICAO(substr);
        }

        if (count == 6)
        {
            // Check for valid latitude
            if (substr == "")
            {
                airport_valid = false;
                break;
            }
            else
            {
                double latitude = std::stold(substr);
                if (latitude > 90 || latitude < -90)
                {
                    airport_valid = false;
                    break;
                }
                airport -> setLatitude(latitude);
            }
        }

        if (count == 7)
        {
            // Check for valid longtitude
            if (substr == "")
            {
                airport_valid = false;
                break;
            }
            else
            {
                double longitude = std::stold(substr);
                if (longitude > 180 || longitude < -180)
                {
                    airport_valid = false;
                    break;
                }
                airport -> setLongitude(longitude);
            }
        }
        count ++;
    }

    // Determine whether the airport should be stored
    if (airport_valid == true)
    {
        /* airport_vec.size() is the actual index refers to the airport,
            which should equal to the airport assigned id. (not the original
            one if there exist invalid airports)

            The actual assigned airport id is 'airport_vec.size()'
        */
        airport -> setId(airport_vec.size());
        id_change[orig_id] = airport_vec.size();

        // Set the two maps (IATA -> ID AND ICAO -> ID)
        if (airport -> getIATA() != "")
        {
            iata_id[airport -> getIATA()] = airport -> getID();
        }
        if (airport -> getICAO() != "")
        {
            icao_id[airport -> getICAO()] = airport -> getID();
        }

        /* Get the airport actual id from user input airport */
        if (airport_1 == airport -> getName())
        {
            airport_1_id = airport -> getID();
        }
        else if (airport_2 == airport -> getName())
        {
            airport_2_id = airport -> getID();
        }

        airport_vec.push_back(*airport);
    }
    else
    {
        delete airport;
    }
}


/**
    This is the overloading function of parse_correct_airport for test cases
**/
void Readfile::parse_correct_airport(string s, vector<Airport> & airport_vec)
{
    // Construct an airport object
    Airport *airport = new Airport();

    stringstream s_stream(s);
    int count = 0;
    bool airport_valid = true;
    unsigned int orig_id = 0;
    while(s_stream.good())
    {
        // 'substr' is the string of every parse (different portions of the line)
        string substr;
        getline(s_stream, substr, ',');
        
        if (count == 0)
        {
            // Read the airport's original id
            orig_id = std::stoul(substr);
        }

        if (count == 1)
        {
            // Check for valid airport name
            if (substr == "")
            {
                airport_valid = false;
                break;
            }

            if (substr[0] != '\"' || substr[substr.length() - 1] != '\"')
            {
                airport_valid = false;
                break;
            }

            airport -> setName(substr);
        }

        if (count == 2)
        {
            // Check for valid city name
            if (substr == "")
            {
                airport_valid = false;
                break;
            }

            if (substr[0] != '\"' || substr[substr.length() - 1] != '\"')
            {
                airport_valid = false;
                break;
            }

            airport -> setCity(substr);
        }

        if (count == 3)
        {
            // Check for valid country name
            if (substr == "")
            {
                airport_valid = false;
                break;
            }

            if (substr[0] != '\"' || substr[substr.length() - 1] != '\"')
            {
                airport_valid = false;
                break;
            }

            airport -> setCountry(substr);
        }

        if (count == 4)
        {
            // Add IATA (IATA can be \N if not assigned)
            airport -> setIATA(substr);
        }

        if (count == 5)
        {
            // Add ICAO (ICAO can be \N if not assigned)
            airport -> setICAO(substr);
        }

        if (count == 6)
        {
            // Check for valid latitude
            if (substr == "")
            {
                airport_valid = false;
                break;
            }
            else
            {
                long double latitude = std::stold(substr);
                if (latitude > 90 || latitude < -90)
                {
                    airport_valid = false;
                    break;
                }
                airport -> setLatitude(latitude);
            }
        }

        if (count == 7)
        {
            // Check for valid longtitude
            if (substr == "")
            {
                airport_valid = false;
                break;
            }
            else
            {
                long double longitude = std::stold(substr);
                if (longitude > 180 || longitude < -180)
                {
                    airport_valid = false;
                    break;
                }
                airport -> setLongitude(longitude);
            }
        }
        count ++;
    }

    // Determine whether the airport should be stored
    if (airport_valid == true)
    {
        /* airport_vec.size() is the actual index refers to the airport,
            which should equal to the airport assigned id. (not the original
            one if there exist invalid airports)

            The actual assigned airport id is 'airport_vec.size()'
        */
        airport -> setId(airport_vec.size());
        id_change[orig_id] = airport_vec.size();

        // Set the two maps (IATA -> ID AND ICAO -> ID)
        if (airport -> getIATA() != "")
        {
            iata_id[airport -> getIATA()] = airport -> getID();
        }
        if (airport -> getICAO() != "")
        {
            icao_id[airport -> getICAO()] = airport -> getID();
        }
        airport_vec.push_back(*airport);
    }
    else
    {
        delete airport;
    }
}


/**
    parse_correct_routes is the helper function that parses strings from every line of the file, and
    checks for data validity. 

    @param routes_vec The vector that stores every Route struct
    @param s Every line of the input file
**/
void Readfile::parse_correct_routes(string s, vector<Route*> & routes_vec, vector<Airport> & airport_vec)
{
    // Instantiate a Route struct
    Route *route = new Route(0, 0, 0);

    stringstream s_stream(s);
    int count = 0;
    bool route_valid = true;
    string source;
    string dest;
    while(s_stream.good())
    {
        // 'substr' is the string of every parse (different portions of the line)
        string substr;
        getline(s_stream, substr, ',');

        if (count == 2)
        {
            // Read source Airport IATA or ICAO
            source = substr;
        }
        
        if (count == 3)
        {
            // Check for valid source airport ID
            if (substr == "")
            {
                route_valid = false;
                break;
            }
            else if (isNumber(substr) == 0)
            {
                // Find the corresponding airport ID
                if (source == "" || source == "\\N")
                {
                    // Not enough information to find the source aiport, discard the route
                    route_valid = false;
                    break;
                }

                if (source.length() == 3)
                {
                    string key = "\"" + source + "\"";
                    if (iata_id.find(key) == iata_id.end())
                    {
                        // The source airport does not exist, discard the route
                        route_valid = false;
                        break;
                    }
                    else
                    {
                        route -> setStartID(iata_id[key]);
                    }
                }
                else
                {
                    string key = "\"" + source + "\"";
                    if (icao_id.find(key) == icao_id.end())
                    {
                        // The source airport does not exist, discard the route
                        route_valid = false;
                        break;
                    }
                    else
                    {
                        route -> setStartID(icao_id[key]);
                    }
                }
            }
            else
            {
                if (id_change.find(std::stoul(substr)) == id_change.end())
                {
                    // The source airport does not exist, discard the route
                    route_valid = false;
                    break;
                }
                else
                {
                    route -> setStartID(id_change[std::stoul(substr)]);
                }   
            }
        }

        if (count == 4)
        {
            // Read destination Aiport IATA or ICAO
            dest = substr;
        }

        if (count == 5)
        {
            // Check for valid destination airport ID
            if (substr == "")
            {
                route_valid = false;
                break;
            }
            else if (isNumber(substr) == 0)
            {
                // Find the corresponding airport ID
                if (dest == "" || dest == "\\N")
                {
                    // Not enough information to find the destination airport, discard the route
                    route_valid = false;
                    break;
                }

                if (dest.length() == 3)
                {
                    string key = "\"" + dest + "\"";
                    if (iata_id.find(key) == iata_id.end())
                    {
                        // The destination airport does not exist, discard the route
                        route_valid = false;
                        break;
                    }
                    else
                    {
                        route -> setEndID(iata_id[key]);
                    }
                }
                else
                {
                    string key = "\"" + dest + "\"";
                    if (icao_id.find(key) == icao_id.end())
                    {
                        // The destination airport does not exist, discard the route
                        route_valid = false;
                        break;
                    }
                    else
                    {
                        route -> setEndID(icao_id[key]);
                    }
                }
            }
            else
            {
                if (id_change.find(std::stoul(substr)) == id_change.end())
                {
                    // The destination airport does not exist, discard the route
                    route_valid = false;
                    break;
                }
                else
                {
                    route -> setEndID(id_change[std::stoul(substr)]);
                }   
            }
        }
        count ++;
    }

    // Determine whether the route should be stored
    if (route_valid == true)
    {
        routes_vec.push_back(route);

        // Calculate distance between start airport and destination airport
        long double lat_1 = airport_vec[route -> getStartID()].getLatitude();
        long double long_1 = airport_vec[route -> getStartID()].getLongitude();
        long double lat_2 = airport_vec[route -> getEndID()].getLatitude();
        long double long_2 = airport_vec[route -> getEndID()].getLongitude();

        route -> setDist(distance(lat_1, long_1, lat_2, long_2));
    }
    else
    {
        delete route;
    }
}
