# yuli9-yixiaof2-nankang2-zisuj2
Final project for team: dhrev

## Dijkstra’s Algorithm:

We use Dijkstra’s Algorithm to find the shortest path from one airport to the other. When a user enters the departure and destination airports’ ID, it will return the shortest path in terms of a vector containing airports’ ID, and the total distance will be renewed. If there is no path between, it should return a size one vector with 999999, and the total distance is 0. Consider the case that the straight line between two airports is always the shortest path in real life, we adopt the priority queue built in c++, since we don't need to reset the value stored in each node.

To test Dijkstra’s Algorithm, we used a small dataset. The dataset can be found at tests->Dij_test_Airport1.txt, Dij_test_Route1.txt. Here is the graph representing this dataset:
![image](https://github-dev.cs.illinois.edu/cs225-sp22/yuli9-yixiaof2-nankang2-zisuj2/blob/main/IMG/test_Dijkstra1.jpg)

We also visualize the graph by printing each airport’s adjacent airports.
![image](https://github-dev.cs.illinois.edu/cs225-sp22/yuli9-yixiaof2-nankang2-zisuj2/blob/main/IMG/test_Dijkstra2.png)

We use many test cases to test: 1. no path between; 2. has a path with one edge; 3. no path but has an opposite direction path; 4. has two paths but the one with three edges is shorter than the one with two edges. Here is the result of these test cases, which are all as our expected.
![image](https://github-dev.cs.illinois.edu/cs225-sp22/yuli9-yixiaof2-nankang2-zisuj2/blob/main/IMG/test_Dijkstra3.png)

## PageRank



## BFS

We use the BFS as our traversal method. The algorithm is able to run through the full-scaled dataset and visit every nodes in the graph, which are airports in our case. The declaration of BFS is in the Graph.h, and the implementation is in the Graph.cpp. Since we have a large dataset, and the result of BFS is visitng every node, so it's hard to visualize the result by printing out the airport information on the terminal. Therefore we tried to plot the dots on a world map to visualize the result.

The visulization of world map after runnign BFS with the full-scaled dataset from test.cpp:
![image](https://github-dev.cs.illinois.edu/cs225-sp22/yuli9-yixiaof2-nankang2-zisuj2/blob/main/IMG/BFS_output.png)

Before we run the BFS on the large dataset, we had tested it with a small arbitrary set of information. With some small data, either connected or unconnected, the algorithm is able to traverse all the nodes and print it out.