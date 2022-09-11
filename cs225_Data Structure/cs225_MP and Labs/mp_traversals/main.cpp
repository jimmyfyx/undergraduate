
#include "cs225/PNG.h"
#include "FloodFilledImage.h"
#include "Animation.h"

#include "imageTraversal/DFS.h"
#include "imageTraversal/BFS.h"

#include "colorPicker/RainbowColorPicker.h"
#include "colorPicker/GradientColorPicker.h"
#include "colorPicker/GridColorPicker.h"
#include "colorPicker/SolidColorPicker.h"
#include "colorPicker/MyColorPicker.h"

using namespace cs225;

int main() {

  // @todo [Part 3]
  // - The code below assumes you have an Animation called `animation`
  // - The code provided below produces the `myFloodFill.png` file you must
  //   submit Part 3 of this assignment -- uncomment it when you're ready.

  // Input an image
  PNG image = PNG();
  image.readFromFile("i.png"); 

  Point start = Point(40, 40);
  DFS dfs = DFS(image, start, 0.05);
  BFS bfs = BFS(image, start, 0.05);
  MyColorPicker my = MyColorPicker(0);
  FloodFilledImage image_ = FloodFilledImage(image);
  image_.addFloodFill(dfs, my);
  image_.addFloodFill(bfs, my);
  Animation animation = image_.animate(1000);

  PNG lastFrame = animation.getFrame( animation.frameCount() - 1 );
  lastFrame.writeToFile("myFloodFill.png");
  animation.write("myFloodFill.gif");
  
  return 0;
}
