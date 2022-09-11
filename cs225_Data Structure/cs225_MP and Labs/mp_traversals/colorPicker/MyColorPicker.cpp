#include "../cs225/HSLAPixel.h"
#include "../Point.h"

#include "ColorPicker.h"
#include "MyColorPicker.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace cs225;

/**
 * Picks the color for pixel (x, y).
 * Using your own algorithm
 */
MyColorPicker::MyColorPicker(double hue)
{
  hue_ = hue;
}

HSLAPixel MyColorPicker::getColor(unsigned x, unsigned y) {
  /* @todo [Part 3] */
  srand(time(0));
  hue_ = rand() % 360;
  HSLAPixel pixel(hue_, 0.5, 0.8);
 
  return pixel;
}
