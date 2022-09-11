/**
 * @file common_words.cpp
 * Implementation of the CommonWords class.
 *
 * @author Zach Widder
 * @date Fall 2014
 */

#include "common_words.h"

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

using std::string;
using std::vector;
using std::ifstream;
using std::cout;
using std::endl;
using std::feof;

string remove_punct(const string& str)
{
    string ret;
    std::remove_copy_if(str.begin(), str.end(), std::back_inserter(ret),
                        [](int c) {return std::ispunct(c);});
    return ret;
}

CommonWords::CommonWords(const vector<string>& filenames)
{
    // initialize all member variables
    init_file_word_maps(filenames);
    init_common();
}

void CommonWords::init_file_word_maps(const vector<string>& filenames)
{
    // make the length of file_word_maps the same as the length of filenames
    file_word_maps.resize(filenames.size());

    // go through all files
    for (size_t i = 0; i < filenames.size(); i++) {
        // get the corresponding vector of words that represents the current
        // file
        vector<string> words = file_to_vector(filenames[i]);
        map<string, unsigned int> word_freq;
        // Traverse the words vector
        for (unsigned int j = 0; j < words.size(); j ++)
        {
            if (word_freq.find(words[j]) == word_freq.end())
            {
                // Add a new word to the map
                word_freq[words[j]] = 1;
            }
            else
            {
                // Increment the frequency
                word_freq[words[j]] = word_freq[words[j]] + 1;
            }
        }
        file_word_maps[i] = word_freq;
    }
}

void CommonWords::init_common()
{
    // Traverse file_word_maps
    for (unsigned int i = 0; i < file_word_maps.size(); i ++)
    {
        // Traverse each map
        for (auto it = file_word_maps[i].begin(); it != file_word_maps[i].end(); it ++)
        {
            if (common.find(it -> first) == common.end())
            {
                common[it -> first] = 1;
            }
            else
            {
                common[it -> first] = common[it -> first] + 1;
            }
        }
    }
}

/**
 * @param n The number of times to word has to appear.
 * @return A vector of strings. The vector contains all words that appear
 * in each file >= n times.
 */
vector<string> CommonWords::get_common_words(unsigned int n) const
{
    vector<string> out;
    // Traverse the common map
    for (auto it = common.begin(); it != common.end(); it ++)
    {
        string word_ = it -> first;
        // Check the word appears in each file
        if (it -> second >= file_word_maps.size())
        {
            // Check the frequency of the word in each file
            bool word_valid = true;
            for (unsigned int i = 0; i < file_word_maps.size(); i ++)
            {
                if (file_word_maps[i].at(word_) < n)
                {
                    word_valid = false;
                    break;
                }
            }

            if (word_valid == true)
            {
                out.push_back(word_);
            }
        }
    }
    
    return out;
}

/**
 * Takes a filename and transforms it to a vector of all words in that file.
 * @param filename The name of the file that will fill the vector
 */
vector<string> CommonWords::file_to_vector(const string& filename) const
{
    ifstream words(filename);
    vector<string> out;

    if (words.is_open()) {
        std::istream_iterator<string> word_iter(words);
        while (!words.eof()) {
            out.push_back(remove_punct(*word_iter));
            ++word_iter;
        }
    }
    return out;
}
