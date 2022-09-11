# Final Project Proposal

1. **Leading Question:** 
The OpenFlight dataset is the dataset we decide to use for our project. Given some useful information about airports, and routes, the primary problem we want to tackle is the optimal flight route a person can take between any two given points on Earth. Here, we define optimal to be taking the shortest travel time dependent of the total distance and the number of stops. This is a function that has been achieved by many travel planning applications and digital maps, and the principal goal is to make people’s travel more convenient. 

2. **Dataset Acquisition and Processing:**
- Data format: The two major databases we consider using are the airport database and route database. Each database is divided into lines of string, and each line contains multiple pieces of information separated by commas in a specific order. 
    - Summary of useful information for now:
        - Airport: ID, Name, City, Country, Latitude, Longitude
        - Route: Source Airport ID, Destination Airport ID, Stops
We plan to set each database in a single .txt file and read the two files. When reading the file, needed information will be stored. We decided to use the full database, which contains 14110 airports and 67663 airline routes. 
- Data Correction: As described above, we will parse each line in the database and store the data in the way described below. We will check if airport’s latitude and longitude are out of range (ie. range for latitude [-90, 90], range for longitude[0, 360)) whenever the data is read initially from the file. 
- Data Storage:
    - Use objects created from the class Point<3> to store the coordinate (x, y, z) of each airport, where (x, y, z) is converted from Latitude and Longitude.
    - Use vectors to store all the Point objects.
    - Use objects created from the class Airport to store information of each airport, including Name, ID, and Latitude, Longitude.
    - Use a 2d vector to store adjacency lists of all airports in the dataset, also the vertices in the graph we build, where the non-zero values are replaced by the weight of the corresponding edge.
    - Use map to store all the Airport objects and its corresponding coordinate, where the key is object Point, and the value is object Airport.
    - Use KDTree to store the airport’s location information, i.e., Point.

- Estimate of the total storage costs: since we would use a 2D vector, the estimated storage would be O(n^2) given n entries in the airport dataset.

3. **Graph Algorithms:**
- Graph Traversal Algorithms: DFS
Function Inputs: Base PNG graph, start vertex.
Function Outputs: void
Function Efficiency: O(n) because all traversal takes O(n) time
Purpose: We use DFS to traverse through the graph of airports. We want to map all the locations of the airports and routes among them on the base graph by visiting all the vertices and edges. Since we want to map all the edges too, DFS is the appropriate traversal algorithm.
- Covered Algorithm: Dijkstra's algorithm
Function Inputs: Departure airport coordinates, graph
Function Outputs: Arrival airport coordinates
Function Efficiency: O(v^2), which v is the number of vertices
Purpose: This algorithm finds the nearest path between two airports. This includes all the edges and airports connected between them since it’s possible for two airports not having a direct connection.
- Complex or Uncovered: PageRank
Function Inputs: Directed graph
Function Outputs: Weight of airports
Function Efficiency: O(n+m), n is the number of nodes(airports), m is the number of edges(routes)
Purpose: Find the most popular airports, which means those airports have the most departure/arrival routes. We want to use this algorithm to label more significant airports compared to less significant airports.

4. **Timeline:**
- Code Implementation:
    - Task 1: Helper Functions: eg. convert airport location(latitude, longitude) to xyz coordinate
        - Expected time period and DDL: Mar.26 - Apr.3 (1 week)
    - Task 2: Parse data and build data structures
        - Expected time period and DDL: Mar.26 - Apr. 3 (1 week)
    - Task 3: Implement DFS
        - Use DFS to print a part of data to check data storage
        - Expected time period and DDL: Apr.4 - Apr.10 (1 week)
    - Task 4: Dijkstra's algorithm
        - Expected time period and DDL: Apr.11 - Apr.24 (2 weeks)
    - Task 5: PageRank
        - Expected time period and DDL: Apr.11 - Apr.24 (2 weeks)
    - Task 6: Visualization
        - Expected time period and DDL: Apr.25 - May.1 (1 week)

- Logistics and Others:
    - Ultimate README.md:
        - Expected time period and DDL: May.1 - May.5 (5 days)
    - Written Report (results.md):
        - Expected time period and DDL: Analyze the output and correctness of each algorithm immediately after it is completed.
    - Preparation of Final Presentation:
        - Expected time period and DDL: May.1 - May.5 (5 days)


