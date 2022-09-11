#include "Image.h"
#include "StickerSheet.h"

int main() {
  // Read the base image
  Image base;
  base.readFromFile("alma.png");

  // Set up a new Stickersheet
  StickerSheet stickersheet = StickerSheet(base, 5);

  // Read three sticker
  Image sticker_1;
  sticker_1.readFromFile("i.png");
  Image sticker_2;
  sticker_2.readFromFile("i.png");
  Image sticker_3;
  sticker_3.readFromFile("i.png");

  
  // Add the sticker to the stickersheet
  int sticker_1_layer = stickersheet.addSticker(sticker_1, 20, 200);
  int sticker_2_layer = stickersheet.addSticker(sticker_2, 100, 200);
  int sticker_3_layer = stickersheet.addSticker(sticker_3, 180, 200);

  // Save the resulting stickersheet
  Image output = stickersheet.render();
  base = output;
  base.writeToFile("myImage.png");
  
  return 0;
}
