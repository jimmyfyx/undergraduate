/**
 * @file cartalk_puzzle.cpp
 * Holds the function which solves a CarTalk puzzler.
 *
 * @author Matt Joras
 * @date Winter 2013
 */

#include <fstream>

#include "cartalk_puzzle.h"

using namespace std;

/**
 * Solves the CarTalk puzzler described here:
 * http://www.cartalk.com/content/wordplay-anyone.
 * @return A vector of (string, string, string) tuples
 * Returns an empty vector if no solutions are found.
 * @param d The PronounceDict to be used to solve the puzzle.
 * @param word_list_fname The filename of the word list to be used.
 */
vector<std::tuple<std::string, std::string, std::string>> cartalk_puzzle(PronounceDict d,
                                    const string& word_list_fname)
{
    vector<std::tuple<std::string, std::string, std::string>> ret;

    ifstream wordsFile(word_list_fname);
    string word;
    if (wordsFile.is_open()) 
    {
        /* Reads a line from `wordsFile` into `word` until the file ends. */
        while (getline(wordsFile, word)) 
        {
            if (word.length() >= 3)
            {
                // Check the word itself and the word with moving the first letter
                bool first_second = false;
                string substring_1 = word.substr(1);
                first_second = d.homophones(word, substring_1);

                // Check the word with moving the first letter and the word with moving the second letter
                bool second_third = false;
                string substring_2 = word[0] + word.substr(2);
                second_third = d.homophones(substring_1, substring_2);

                if (first_second == true && second_third == true)
                {
                    // Push to vector
                    tuple <string, string, string> tuple_ = {word, substring_1, substring_2};
                    // auto tuple_ = std::make_tuple(word, substring_1, substring_2);
                    ret.push_back(tuple_);
                }
            }
        }
    }
    return ret;
}
