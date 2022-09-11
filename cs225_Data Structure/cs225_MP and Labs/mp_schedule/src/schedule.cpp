/**
 * @file schedule.cpp
 * Exam scheduling using graph coloring
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <map>
#include <unordered_map>

#include "schedule.h"
#include "utils.h"
#include <algorithm>

using namespace std;


Graph::Graph()
{
    /* Nothing */
}


Graph::~Graph()
{
    /* Nothing */
}


Graph::Graph(V2D roster)
{
    /* Initialize the adjacency matrix */
    for (unsigned int i = 0; i < roster.size(); i ++)
    {
        vector<int> row_vec;
        for (unsigned int j = 0; j < roster.size(); j ++)
        {
            row_vec.push_back(0);
        }
        adj_matrix.push_back(row_vec);
    }


    for (unsigned int i = 0; i < roster.size(); i ++)
    {
        string course_name = roster[i][0];
        for (unsigned int j = 1; j < roster[i].size(); j ++)
        {
            /* For each student, check whether also in other courses */
            string student_name = roster[i][j];
            for (unsigned int m = 0; m < roster.size(); m ++)
            {
                if (roster[m][0] != course_name)
                {
                    for (unsigned int n = 1; n < roster[m].size(); n ++)
                    {
                        if (roster[m][n] == student_name)
                        {
                            /* Two courses share a same student */
                            adj_matrix[i][m] = 1;
                            break;
                        }
                    }
                }
            }
        }
    }
    // cout << "adj_matrix size: " << adj_matrix.size() << endl;
}



void Graph::printAdjMatrix()
{
    for (unsigned int i = 0; i < adj_matrix.size(); i ++)
    {
        for (unsigned int j = 0; j < adj_matrix[i].size(); j ++)
        {
        cout << adj_matrix[i][j] << " ";
        }
        cout << "\n" << endl;
    }
}



V2D Graph::GraphColoring(V2D final_roster, vector<string> timeslots)
{
    /* Try different starting vertex (course) */
    for (unsigned int a = 0; a < final_roster.size(); a ++)
    {
        bool valid_sequence = true;

        /** 
            The vector assigns a timeslot for a course.
            Index as course, and value as timeslot (all in index form)
        **/
        vector<int> result;
        /* Initialize the result vector and assign a time slot for the first course */
        for (unsigned int i = 0; i < final_roster.size(); i ++)
        {
            /* -1 denotes that a course has not been assigned a timeslot */
            result.push_back(-1);
        }
        result[a] = 0;

        /**
            The vector indicates whether the color is available for the current course.
            Index as color, and values as the availability of the color.
            'false' indicates the color is not available for the current course.

            This will be a temporary vector in use when arriving at every vertex of the graph
        **/
        vector<bool> available_color;
        for (unsigned int i = 0; i < timeslots.size(); i ++)
        {
            available_color.push_back(true);
        }


        /* Assign timeslots to the remainning courses */
        for (unsigned int i = 0; i < final_roster.size(); i ++)
        {
            if (i != a)
            {
                /* Check adjacent vertices (courses) and update the available_color vector */
                for (unsigned int j = 0; j < adj_matrix[i].size(); j ++)
                {
                    if (adj_matrix[i][j] == 1)
                    {
                        if (result[j] != -1)
                        {
                            available_color[result[j]] = false;
                        }
                    }
                }

                /* Find the available color for the current course */
                bool find_color = false;
                for (unsigned int m = 0; m < available_color.size(); m ++)
                {
                    if (available_color[m] == true)
                    {
                        /* Find an available color */
                        result[i] = m;
                        find_color = true;
                        break;
                    }
                }

                if (find_color == false)
                {
                    /* Fail to construct a schedule for the sequence, break from the current sequence */
                    valid_sequence = false;
                    break;
                }

                /* Reset the available color vector for the use of the next course */
                for (unsigned int n = 0; n < available_color.size(); n ++)
                {
                    available_color[n] = true;
                }
            } 
        }

        /* Check whether the current sequence is valid */
        if (valid_sequence == true)
        {
            /** Save the sequence and return **/
            /* Initialize the return vector */
            vector<vector<string>> ret;
            for (unsigned int k = 0; k < timeslots.size(); k ++)
            {
                vector<string> timeslot_;
                timeslot_.push_back(timeslots[k]);
                ret.push_back(timeslot_);
            }

            for (unsigned int k = 0; k < result.size(); k ++)
            {
                ret[result[k]].push_back(final_roster[k][0]);
            }
            
            return ret;
        }
    }

    /** Cannot find a valid schedule **/
    vector<vector<string>> invalid;
    vector<string> row;
    row.push_back(std::to_string(-1));
    invalid.push_back(row);
    return invalid;
}



/**
 * Takes a filename and reads in all the text from the file
 * Newline characters are also just characters in ASCII
 * 
 * @param filename The name of the file that will fill the string
 */
std::string file_to_string(const std::string& filename){
  std::ifstream text(filename);

  std::stringstream strStream;
  if (text.is_open()) {
    strStream << text.rdbuf();
  }
  return strStream.str();
}

/**
 * Given a filename to a CSV-formatted text file, create a 2D vector of strings where each row
 * in the text file is a row in the V2D and each comma-separated value is stripped of whitespace
 * and stored as its own string. 
 * 
 * Your V2D should match the exact structure of the input file -- so the first row, first column
 * in the original file should be the first row, first column of the V2D.
 *  
 * @param filename The filename of a CSV-formatted text file. 
 */
V2D file_to_V2D(const std::string & filename){
  // First convert the whole file to a string
  string file = file_to_string(filename);
  // Split each row in the file
  vector<string> row_string;
  int num_rows = SplitString(file,'\n',row_string);

  // Initialize the 2D vector
  vector<vector<string>> vector_2D;
  for (int i = 0; i < num_rows; i ++)
  {
    // Parse each row
    vector<string> row;
    int num_cols = SplitString(row_string[i], ',', row);
    for (int j = 0; j < num_cols; j ++)
    {
      if (row[j][0] == ' ' && row[j][row[j].length() - 1] == ' ')
      {
        row[j] = Trim(row[j]);
      }
      else if (row[j][0] == ' ' && row[j][row[j].length() - 1] != ' ')
      {
        row[j] = TrimLeft(row[j]);
      }
      else if (row[j][0] != ' ' && row[j][row[j].length() - 1] == ' ')
      {
        row[j] = TrimRight(row[j]);
      }
    }
    vector_2D.push_back(row);
  }
  return vector_2D;
}

/**
 * Given a course roster and a list of students and their courses, 
 * perform data correction and return a course roster of valid students (and only non-empty courses).
 * 
 * A 'valid student' is a student who is both in the course roster and the student's own listing contains the course
 * A course which has no students (or all students have been removed for not being valid) should be removed
 * 
 * @param cv A 2D vector of strings where each row is a course ID followed by the students in the course
 * @param student A 2D vector of strings where each row is a student ID followed by the courses they are taking
 */
V2D clean(V2D & cv, V2D & student){
  // Make a copy of the current class roster
  vector<vector<string>> final_roster;
  for (unsigned int i = 0; i < cv.size(); i ++)
  {
    final_roster.push_back(cv[i]);
  }
  
  // Start to traverse the class roster in row order
  int count_0 = 0;
  for (auto it_0 = cv.begin(); it_0 != cv.end(); it_0 ++)
  {
    if (final_roster[count_0].size() == 1)
    {
      // No stduent in the course originally, delete the course (row)
      auto remove_it_0 = final_roster.begin() + count_0;
      final_roster.erase(remove_it_0);
      continue;
    }

    string course_number = (*it_0)[0];
    int count_1 = 0;
    for (auto it_1 = (*it_0).begin() + 1; it_1 != (*it_0).end(); it_1 ++)
    {
      string student_name = *it_1;
      // Check for the student roster
      bool student_name_exist = false;
      for (auto it_2 = student.begin(); it_2 != student.end(); it_2 ++)
      {
        if ((*it_2)[0] == student_name)
        {
          student_name_exist = true;
          // Check whether the student is enrolled in the course
          bool enrolled = false;
          for (auto it_3 = (*it_2).begin() + 1; it_3 != (*it_2).end(); it_3 ++)
          {
            if (*it_3 == course_number)
            {
              enrolled = true;
              break;
            }
          }

          if (enrolled == false)
          {
            // Revise the final roster (remove the student from the course)
            auto remove_it_1 = final_roster[count_0].begin() + 1 + count_1;
            final_roster[count_0].erase(remove_it_1);
            count_1 --;

            if (final_roster[count_0].size() == 1)
            {
              // No stduent in the course, delete the course (row)
              auto remove_it_0 = final_roster.begin() + count_0;
              final_roster.erase(remove_it_0);
              count_0 --;
            }
          } 
        }
      }

      if (student_name_exist == false)
      {
        // Delete the student's name
        auto remove_it_1 = final_roster[count_0].begin() + 1 + count_1;
        final_roster[count_0].erase(remove_it_1);
        count_1 --;

        // cout << final_roster[count_0].size() << endl;
        if (final_roster[count_0].size() == 1)
        {
          // No stduent in the course, delete the course (row)
          auto remove_it_0 = final_roster.begin() + count_0;
          final_roster.erase(remove_it_0);
          count_0 --;
        }
      } 
      count_1 ++;
    }
    count_0 ++;
  }

  /*
  for (unsigned int i = 0; i < final_roster.size(); i ++)
  {
    cout << "Row Size: " << final_roster[i].size() << endl;
    for (unsigned int j = 0; j < final_roster[i].size(); j ++)
    {
      cout << final_roster[i][j] << " ";
    }
    cout << "\n" << endl;
  }
  */

  return final_roster;
}

/**
 * Given a collection of courses and a list of available times, create a valid scheduling (if possible).
 * 
 * A 'valid schedule' should assign each course to a timeslot in such a way that there are no conflicts for exams
 * In other words, two courses who share a student should not share an exam time.
 * Your solution should try to minimize the total number of timeslots but should not exceed the timeslots given.
 * 
 * The output V2D should have one row for each timeslot, even if that timeslot is not used.
 * 
 * As the problem is NP-complete, your first scheduling might not result in a valid match. Your solution should 
 * continue to attempt different schedulings until 1) a valid scheduling is found or 2) you have exhausted all possible
 * starting positions. If no match is possible, return a V2D with one row with the string '-1' as the only value. 
 * 
 * @param courses A 2D vector of strings where each row is a course ID followed by the students in the course
 * @param timeslots A vector of strings giving the total number of unique timeslots
 */
V2D schedule(V2D courses, std::vector<std::string> timeslots){
  Graph graph_ = Graph(courses);
  V2D outSched = graph_.GraphColoring(courses, timeslots);
  return outSched;
}