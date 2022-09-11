#include "cs225/PNG.h"
#include <list>
#include <iostream>

#include "colorPicker/ColorPicker.h"
#include "imageTraversal/ImageTraversal.h"

#include "Point.h"
#include "Animation.h"
#include "FloodFilledImage.h"

using namespace cs225;

/**
 * Constructs a new instance of a FloodFilledImage with a image `png`.
 * 
 * @param png The starting image of a FloodFilledImage
 */
FloodFilledImage::FloodFilledImage(const PNG & png) {
  /** @todo [Part 2] */
  image = png;
}

/**
 * Adds a FloodFill operation to the FloodFillImage.  This function must store the operation,
 * which will be used by `animate`.
 * 
 * @param traversal ImageTraversal used for this FloodFill operation.
 * @param colorPicker ColorPicker used for this FloodFill operation.
 */
void FloodFilledImage::addFloodFill(ImageTraversal & traversal, ColorPicker & colorPicker) {
  /** @todo [Part 2] */
  trav_vec.push_back(&traversal);
  color_vec.push_back(&colorPicker);
}

/**
 * Creates an Animation of frames from the FloodFill operations added to this object.
 * 
 * Each FloodFill operation added by `addFloodFill` is executed based on the order
 * the operation was added.  This is done by:
 * 1. Visiting pixels within the image based on the order provided by the ImageTraversal iterator and
 * 2. Updating each pixel to a new color based on the ColorPicker
 * 
 * While applying the FloodFill to the image, an Animation is created by saving the image
 * after every `frameInterval` pixels are filled.  To ensure a smooth Animation, the first
 * frame is always the starting image and the final frame is always the finished image.
 * 
 * (For example, if `frameInterval` is `4` the frames are:
 *   - The initial frame
 *   - Then after the 4th pixel has been filled
 *   - Then after the 8th pixel has been filled
 *   - ...
 *   - The final frame, after all pixels have been filed)
 */ 
Animation FloodFilledImage::animate(unsigned frameInterval) const {
  Animation animation;
  PNG copy = image;
  for (unsigned int i = 0; i < trav_vec.size(); i ++)
  {
    // Execute each operation
    int count = 0;
    // for (it = trav_vec[i].begin(); it < trav_vec.end(); it ++)
    // for (const Point & p: *(trav_vec[i]))
    for (auto it = trav_vec[i] -> begin(); it != trav_vec[i] -> end(); ++it)
    {
      if (count % frameInterval == 0)
      {
        // Add a new frame
        animation.addFrame(copy);
      }

      // For each point, use the colorpicker to change the color of
      // the corresponding pixel
      HSLAPixel & pixel = copy.getPixel((*it).x, (*it).y);
      pixel = color_vec[i] -> getColor((*it).x, (*it).y);
      /*
      const HSLAPixel & pixel_ = color_vec[i] -> getColor(p.x, p.y);
      pixel.h = pixel_.h;
      pixel.s = pixel_.s;
      pixel.l = pixel_.l;
      pixel.a = pixel_.a;
      */

      count ++;
    }
  }
  // Add the finished image
  animation.addFrame(copy);

  return animation;
}
