/**
 * @file maptiles.cpp
 * Code for the maptiles function.
 */

#include <iostream>
#include <map>
#include "maptiles.h"
//#include "cs225/RGB_HSL.h"

using namespace std;


Point<3> convertToXYZ(LUVAPixel pixel) {
    return Point<3>( pixel.l, pixel.u, pixel.v );
}

MosaicCanvas* mapTiles(SourceImage const& theSource,
                       vector<TileImage>& theTiles)
{
    /**
     * @todo Implement this function!
     */

    // Create a new Mosaic canvas
    MosaicCanvas *new_canvas = new MosaicCanvas(theSource.getRows(), theSource.getColumns());

    // Construct a KDTree for the tile images
    map<Point<3>,TileImage*> points_image;
    vector<Point<3>> points;
    for (unsigned int i = 0; i < theTiles.size(); i ++)
    {
        points.push_back(convertToXYZ(theTiles[i].getAverageColor()));
        points_image[convertToXYZ(theTiles[i].getAverageColor())] = & theTiles[i];
    }
    KDTree<3> tree = KDTree<3>(points);

    for (int i = 0; i < theSource.getRows(); i ++)
    {
        for (int j = 0; j < theSource.getColumns(); j ++)
        {
            // For each image region in the source image
            LUVAPixel avg_color = theSource.getRegionColor(i, j);
            Point<3> original = convertToXYZ(avg_color);
            Point<3> target = tree.findNearestNeighbor(original);
            // Find the tile image with the given point
            // and add it to the new canvas
            new_canvas -> setTile(i, j, points_image[target]);
        }
    }

    return new_canvas;
}

