# yuli9-yixiaof2-nankang2-zisuj2
Final project for team: dhrev
## Github Organization
### /proj
All the code stores in proj.
Airport.h/cpp & Route.h/cpp - contain two classes - Airport and Route.
Graph.h/cpp - contain 3 algorithms (BFS, Dijkstra's algorithm and Pagerank) and graph visualization.
Readfile.h/cpp - used to parse the dataset
main.cpp - run all the algorithms and show their output
test.cpp - test the visualization on world map

Most of tests are stored in /proj/tests, apart from Dij_test_big and Dij_test_small which we used to test our Dijkstra's algorithm during development, and test.cpp which is used to debug visualization.

### /proj/tests
Contain all test cases for data parsing and test datasets for those test cases.

### /proj/data
Contain the full scale dataset that we will use in main.cpp

### /proj/IMG
Contain images that will be shown in .md files on github.

### /proj/text_files
All text documents apart from dataset will be stored here. Contains our log, final presentation link, README.md, result.md,proposal, and team contract.

## Running Instructions
### Test cases instruction
To run all the test cases loacated in the /tests folder, first compile by running make test, then ./test to run all the test cases.

### Test cases instruction
To run main.cpp, first run make, then run ./airports. Then, there should be two prompts on the screen that request the user to enter source and destination airport name, which should be any of the airport names in our airport dataset (airport.txt in /data) without double quotes. Next, some messages about shortest path should be shown, and there is another prompt that allows the user to choose how many ranks to print for the PageRank algorithm, which should be the top most important airports in the dataset. 


