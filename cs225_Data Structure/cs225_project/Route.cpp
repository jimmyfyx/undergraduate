 /*
 * @file Route.h
 *
 * @author DHREV
 */

#include "Route.h"

Route::Route(unsigned start, unsigned end, long double dist): 
    startID(start), endID(end), dist_(dist) {}

unsigned Route::getStartID() const
{
    return startID;
}

unsigned Route::getEndID() const
{
    return endID;
}

long double Route::getDist() const
{
    return dist_;
}

void Route::setStartID(unsigned start)
{
    startID = start;
}

void Route::setEndID(unsigned end)
{
    endID = end;
}

void Route::setDist(long double dist)
{
    dist_ = dist;
}