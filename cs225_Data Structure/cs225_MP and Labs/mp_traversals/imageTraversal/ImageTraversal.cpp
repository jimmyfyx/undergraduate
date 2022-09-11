#include <cmath>
#include <iterator>
#include <iostream>

#include "../cs225/HSLAPixel.h"
#include "../cs225/PNG.h"
#include "../Point.h"

#include "ImageTraversal.h"

/**
 * Calculates a metric for the difference between two pixels, used to
 * calculate if a pixel is within a tolerance.
 *
 * @param p1 First pixel
 * @param p2 Second pixel
 * @return the difference between two HSLAPixels
 */
double ImageTraversal::calculateDelta(const HSLAPixel & p1, const HSLAPixel & p2) {
  double h = fabs(p1.h - p2.h);
  double s = p1.s - p2.s;
  double l = p1.l - p2.l;

  // Handle the case where we found the bigger angle between two hues:
  if (h > 180) { h = 360 - h; }
  h /= 360;

  return sqrt( (h*h) + (s*s) + (l*l) );
}

/**
 * Default iterator constructor.
 */
ImageTraversal::Iterator::Iterator()
{
  traversal_ = NULL;
}


ImageTraversal::Iterator::Iterator(ImageTraversal *traversal, PNG & png, double tolerance, Point & start) {
  /** @todo [Part 1] */
  traversal_ = traversal;
  image = png;
  start_ = start;
  tolerance_ = tolerance;
  current_ = traversal_ -> peek();

  isVisited = new bool*[image.width()];
  for (unsigned int i = 0; i < image.width(); i++)
  {
    isVisited[i] = new bool[image.height()];
    for (unsigned int j = 0; j < image.height(); j++)
    {
      isVisited[i][j] = false;
    }
  }
}

/**
 * Iterator increment opreator.
 *
 * Advances the traversal of the image.
 */
ImageTraversal::Iterator & ImageTraversal::Iterator::operator++() {
  /** @todo [Part 1] */
  // Mark the current point as visited
  // visited.push_back(current_);
  isVisited[current_.x][current_.y] = true;
  // Push the neighbor points into the stack if available (R, D, L, U)
  if (current_.x + 1 < image.width() && current_.x + 1 >= 0 && current_.y < image.height() && current_.y >= 0)
  {
    traversal_ -> add(Point(current_.x + 1, current_.y));
    // std::cout << (traversal_ -> peek()).x << " " << (traversal_ -> peek()).y << std::endl;
  }

  if (current_.x < image.width() && current_.x >= 0 && current_.y + 1 < image.height() && current_.y + 1 >= 0)
  {
    traversal_ -> add(Point(current_.x, current_.y + 1));
    // std::cout << (traversal_ -> peek()).x << " " << (traversal_ -> peek()).y << std::endl;
  }

  if (current_.x - 1 >= 0 && current_.x - 1 < image.width() && current_.y >= 0 && current_.y < image.height())
  {
    traversal_ -> add(Point(current_.x - 1, current_.y));
    // std::cout << (traversal_ -> peek()).x << " " << (traversal_ -> peek()).y << std::endl;
  }

  if (current_.x < image.width() && current_.x >= 0 && current_.y - 1 >= 0 && current_.y - 1 < image.height())
  {
    traversal_ -> add(Point(current_.x, current_.y - 1));
    // std::cout << (traversal_ -> peek()).x << " " << (traversal_ -> peek()).y << std::endl;
  }

  while (traversal_ -> empty() == false)
  {
    point = traversal_ -> pop();
    // Check if the point has already been visited
    /*
    int visit = 0;
    for (unsigned int i = 0; i < size; i ++)
    {
      if (visited[i] == point)
      {
        visit = 1;
        break;
      }
    }
    

    int out_tol = 0;
    // Check if the point is within tolerance
    */
    if (isVisited[point.x][point.y] == false && calculateDelta(image.getPixel(point.x, point.y), image.getPixel(start_.x, start_.y)) < tolerance_)
    {
      current_ = point;
      break;
    }
  }

  if (traversal_ -> empty() == true)
  {
    // Traversal ends
    traversal_ = NULL;
  }

  return *this;
}

/**
 * Iterator accessor opreator.
 *
 * Accesses the current Point in the ImageTraversal.
 */
Point ImageTraversal::Iterator::operator*() {
  /** @todo [Part 1] */
  return current_;
}

/**
 * Iterator inequality operator.
 *
 * Determines if two iterators are not equal.
 */
bool ImageTraversal::Iterator::operator!=(const ImageTraversal::Iterator &other) {
  /** @todo [Part 1] */
  if (traversal_ == NULL && other.traversal_ == NULL)
  {
    return false;
  }
  return true;
}

