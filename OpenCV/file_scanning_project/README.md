# File Scanning with OpenCV
This is a mini-project I implemented with OpenCV, aimed to reproduce similar functions of most file scanning applications on our phones. Given a file sitting on the desk, the program should automatically detect whether there is a file and determine its precise position. Finally, return the picture of the file in bird's-eye view.

All the codes are included in `file_scanning.py`, and there are specific comments for each used function.

## Methods (High-Level Idea)
  - Since the file is always in rectangle shape, we can use this characteristic to find the contour that looks most like a rectangle.
    - If there are multiple rectangles detected, we can differentiate them with the area inside the contours; If the include area is too small, then it can be a file with a very low probability.
  - After selecting the contour representing the file, we can directly apply two methods in OpenCV, `cv.getPerspectiveTransform()` and `cv.warpPerspective()` to get the bird's-eye view of the given area, which is the file. 

## Results
The document I am using is one of the print-out lecture slides from a course I have taken before, named *ENG 298: Road Map to Graduate School*. <br/>
The following three images are detected canny edges, detected contours (with four red dots), and the scanned file corresponding from left to right:
![My Image](./concate.png)
