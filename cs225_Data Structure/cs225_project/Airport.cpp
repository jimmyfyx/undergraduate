#include "Airport.h"

#include <string>
using namespace std;

Airport::Airport()
{
    name_ = "";
    country_ = "";
    city_ = "";
    id_ = 0;
    latitude_ = 0;
    longitude_ = 0;
}

Airport::Airport(string name, string country, string city, unsigned int id, long double latitude, long double longitude)
{
    name_ = name;
    country_ = country;
    city_ = city;
    id_ = id;
    latitude_ = latitude;
    longitude_ = longitude;
}

Airport::~Airport()
{

}

void Airport::setName(string name)
{
    name_ = name;
}

void Airport::setCountry(string country)
{
    country_ = country;
}

void Airport::setCity(string city)
{
    city_ = city;
}

void Airport::setIATA(string iata)
{
    iata_ = iata;
}

void Airport::setICAO(string icao)
{
    icao_ = icao;
}

void Airport::setId(unsigned int id)
{
    id_ = id;
}

void Airport::setLatitude(long double latitude)
{
    latitude_ = latitude;
}

void Airport::setLongitude(long double longitude)
{
    longitude_ = longitude;
}

string Airport::getName() const
{
    return name_;
}

string Airport::getCountry() const
{
    return country_;
}

string Airport::getCity() const
{
    return city_;
}

string Airport::getIATA() const
{
    return iata_;
}

string Airport::getICAO() const
{
    return icao_;
}

unsigned int Airport::getID() const
{
    return id_;
}

long double Airport::getLatitude() const
{
    return latitude_;
}

long double Airport::getLongitude() const
{
    return longitude_;
}