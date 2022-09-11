#include <iostream>
#include "schedule.h"
using namespace std;

int main() {
    V2D roster = file_to_V2D("../tests/data/c10_s50_2_roster_errors.csv");

    V2D student = file_to_V2D("../tests/data/c10_s50_2_students_errors.csv");
    
    V2D out = clean(roster, student);

    /* Print the corrected roster */
    cout << "---------Final Roster: -----------" << endl;
    for (unsigned int i = 0; i < out.size(); i ++)
    {
        for (unsigned int j = 0; j < out[i].size(); j ++)
        {
        cout << out[i][j] << " ";
        }
        cout << "\n" << endl;
    }
    cout << "\n" << endl;

    Graph graph_ = Graph(out);
    cout << "---------Adjacency Matrix: -----------" << endl;
    graph_.printAdjMatrix();


    std::vector<std::string> timeSlots;
    int slots = 3;
    for(int i = 0; i < slots; i++){
        timeSlots.push_back(std::to_string(i));
    }
    V2D outSched = graph_.GraphColoring(out, timeSlots);
    // schedule(out, timeSlots);

    std::cout << "Coloring:" << std::endl;
    std::cout << "{ ";
    for (int i = 0; i < (int) outSched.size(); ++i){
        std::cout << "{ ";
        for(int j = 0; j < (int) outSched[i].size()-1; ++j){
            std::cout << outSched[i][j] << ", ";
        }
        std::cout << outSched[i][outSched[i].size()-1] << "}, \\" << std::endl;
    }
    std::cout << "}" << std::endl;
}