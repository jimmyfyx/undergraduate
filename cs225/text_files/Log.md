## 2022/3/26 Week 0

1. complete and test Point struture
2. Point(Latitude: Deg, negative is South, positive is North; Longitude: Deg, negative is West, positive is East):
3. Axis: x- axis: for Earth's center to (latitute 0, longitude 90E); y- axis: for Earth's center to (latitute 90N, longitude 0); z- axis: for Earth's center to (latitute 0, longitude 0) 
4. use "g++ testPoint.cpp -std=c++11" to complie, use "./a.out" to see the output

## 2022/4/1 Week 1
author: ZJ

### What goals you had set for the week and whether they were accomplished or not
1. Modify our leading question based on the feedback on the proposal. Accomplished.
2. Build a class that stores necessary data structures to build the graph. Accomplished.
3. Build a class that parses given files into entries of airports and routes. Accomplished.

### What specific tasks each member of your team accomplished in the week
**Nan Kang, Yu Li: Modified our leading question**
1. Changed the user input from two arbitrary coordinates to two cities and countries, which better simulates the real situations.
2. Disregarded stops in the routes since waiting time during stops varies from route to route and specific information is not provided.
3. Added another question of finding the most popular airport based on the number of incoming routes using the PageRank algorithm, which can be visulized by the node size each airport takes on the final image.

**Zisu Jiang: Finished data structures of Graph**
1. A vector of bool to mark whether the nodes in the graph is visited during traversals.
2. A 2d vector of double, alse the adjacency list to store distances between any two airports, i.e., the weights of edges.
3. A map that helps to parse the vector of routes into ordered adj list, whose keys are pair of (startID, endID), and values are distance.
4. A pointer to vector of Airports stored outside (main's stack) so we can access info about airports in Airport::DFS.

**Yixiao Fang: Finished interfaces of Readfile**
1. Take possible missing fields in datasets into consideration by adding two more Identifiers of airports and reassign ID if missing in routes.
2. Finish method Readfile::readfile_airport that pushes all valid entries in airports.txt into the vector stored in main's stack.
3. Finish method Readfile::readfile_route that pushes all valid entries in routes.txt into the vector stored in main's stack.

### What problems you encountered (if any) that prevented you from meeting your goals
We did encounter several problems, such as which containers to store necessary data, the trade off between three graph implementations, and data corrections when parsing the files. However, with thorough investigation into online resources, we still accomplished all our goals in the end.

### What you plan to accomplish next week
1. Write bases test cases and makefile.
2. Get started with BFS to print the graph.
3. Modify existed classes if possible.

## 2022/4/8 Week 2
author: YL

### What goals you had set for the week and whether they were accomplished or not
1. Finish BFS in the airport class. Fail to accomplish.
2. Debug the existing class, write test cases. Accomplished.
3. Modifing classes if possible. Accomplished.

### What specific tasks each member of your team accomplished in the week
**Nan Kang : write test cases for airport class and readfile class**
1. Use small-size sample to test the output of readfile and airport.
2. Check if the outputs match with our goal.

**Yixiao Fang: fix bugs in readfile class**
1. Find out bugs in the readfile class.
2. Fix the bugs - not yet accomplished, assigned to next week.

**Zisu Jiang: fix bugs in graph class**
1. Fixed bugs in printGraph and graph constructor.

**Yu Li: write test cases for graph and investigate in BFS**
1. Use arbitrary input to test the graph constructor and printGraph.
2. Investigate BFS - not yet accomplished, assigned to next week.

### What problems you encountered (if any) that prevented you from meeting your goals
We are still arguing about the data storage. The main issue is around the usability of BFS. We are not sure how to incorporate BFS in our implementation, since we don't need traverse the graph to find the shortest path. 

### What you plan to accomplish next week
1. Finish BFS.
2. Debug readFile.
3. test BFS.

## 2022/4/15 Week 3
author: FYX

### What goals you had set for the week and whether they were accomplished or not
1. Finish BFS in the airport class. Accomplished.
2. Debug the Graph class, write unformal test cases. Accomplished.
3. Update the Readfile class with calculating distances between airports. Accomplished.

### What specific tasks each member of your team accomplished in the week
**Nan Kang : Write the first draft of Dijkstra's algorithm and test**
1. Apply the general Dijkstra's algorithm on the graph we build.
2. Use small-size sample to test the output of Dijkstra's algorithm implemented on our graph.

**Yixiao Fang: Update the readfile class and help with implementing the Dijkstra's algorithm**
1. Add the function of calculating distance between two airports with known longitude and latitude.
2. Help apply the general Dijkstra's algorithm on the graph we build.

**Zisu Jiang: Improve the Graph class**
1. Fix bugs in the Graph class, and help testing Dijkstra's algorithm.
2. Start to implement the PageRank algorithm.

**Yu Li: Write BFS**
1. Implement BFS on our graph. 
2. Write tests to check BFS and connectivity of our graph.

### What problems you encountered (if any) that prevented you from meeting your goals
No major problem. 

### What you plan to accomplish next week
1. Go further on PageRank algorithm.
2. Refine the Graph class (including Dijkstra's algorithm).
3. Start to come up with methods for graph visualization. 

## 2022/4/19 Week 4
author: KN

### What goals you had set for the week and whether they were accomplished or not
1. Finish test case for Dijkstra's algorithm. Accomplished.
2. Fix bugs in Readfile.cpp. Accomplished.
3. Start working on PageRank and graph visualization. Started.

### What specific tasks each member of your team accomplished in the week
**Nan Kang : Refine Dijkstra's algorithm and test**
1. Create small-size sample to present the output of Dijkstra's algorithm implemented on our graph.

**Yixiao Fang: Update the readfile class and help with implementing the Dijkstra's algorithm**
1. Edit Readfile.cpp to fix some errors.

**Zisu Jiang: Research on PageRank**
1. Start to implement the PageRank algorithm.

**Yu Li: Research on Visualization**
1. Start to implement graph visualization.

### What problems you encountered (if any) that prevented you from meeting your goals
The running time of Dijkstra's algorithm is too long since the dataset is very big.

### What you plan to accomplish next week
1. Go further on PageRank algorithm.
2. Go further on Graph Visualization.
3. Find a way to shorten the running time of Dijkstra. May reduce dataset's size.


## 2022/4/26 Week 5
author: ZJ

### What goals you had set for the week and whether they were accomplished or not
1. Go further on PageRank algorithm. Accomplished.
2. Go further on Graph Visualization. In Progress (Agreed on no more visualization yet would work on it if possible).
3. Shorten the running time of Dijkstra. In Progress.

### What specific tasks each member of your team accomplished in the week
**Nan Kang**
1. Edited result.md file and added descriptions to Dijkstra. 

**Yixiao Fang**
1. Implemented function of multiplying matrix by vector in Graph.cpp for further use.

**Zisu Jiang**
1. Implemented PageRank algorithm that works on a sample dataset.

**Yu Li**
1. Implemented DrawPoint on a PNG file that may be used in visualization.

### What problems you encountered (if any) that prevented you from meeting your goals
The PageRank algorithm we currently refer to doesn't apply to the adjacency matrix where there is vertex that has no incoming edges or outgoing edges.

### What you plan to accomplish next week
1. Improve PageRank algorithm so it is compatible with the complete dataset.
2. Work on comment of each file and result.md file.
3. Complete text prompts of test programs to polish the final output.

## 2022/5/3 Week 6
author: YL

### What goals you had set for the week and whether they were accomplished or not
1. Completing graph visualization. Accomplished.
2. Completing presentation slides. Accomplished.
3. Record the video. In progress.
4. Wrap up the project. In progress.

### What specific tasks each member of your team accomplished in the week
**Nan Kang**
1. Completing the slides for final presentation. 

**Yixiao Fang**
1. Editing the Readfile.cpp and working on the main function.

**Zisu Jiang**
1. Finishing PageRank algorithm and testing. Perfecting the comments for Graph.

**Yu Li**
1. Finishing the graph implementation, drawLine and draw_rank.

### What problems you encountered (if any) that prevented you from meeting your goals
We reached our goal perfectly this week. Every goal is accomplished ahead of the time
and we have more time to prepare for the final presentation.

### What you plan to accomplish next week
1. Since this is the last week, we don't have any plan for next week.