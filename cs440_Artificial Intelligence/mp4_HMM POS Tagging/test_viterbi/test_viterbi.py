# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math

def main():
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    prediction = []
    # print(initial)
    
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""
    
    """
    implement viterbi algorithm
    """
    test_viterbi = []
    # the first iteration through the test set is used to construct the viterbi structure
    for sentence in test:
        # each sentence owns a viterbi 
        viterbi = []
        # empty = {}
        # viterbi.append(empty)

        # initialize the first column
        first_word = sentence[0]
        first_dict = {}
        for tag in emission:
            # calculate transition prob
            transition_prob = 0
            if tag not in initial:
                transition_prob = initial['UNK']
            else:
                transition_prob = initial[tag]
            # calculate emission prob
            emission_prob = 0
            if first_word not in emission[tag]:
                emission_prob = emission[tag]['UNK']
            else:
                emission_prob = emission[tag][first_word]

            first_dict[tag] = (transition_prob + emission_prob, 'START')
        viterbi.append(first_dict)
        print(viterbi)

        # iterate through the rest of the sentence
        for i in range(1, len(sentence), 1):
            if sentence[i] == 'END':
                break
            word = sentence[i]
            word_dict = {}
            # the word is seen
            for tagB in emission:
                # for each tagB iterate through all the possible previous tags
                # calculate emission prob
                emission_prob = 0
                if word not in emission[tagB]:
                    emission_prob = emission[tagB]['UNK']
                else:
                    emission_prob = emission[tagB][word]

                argmax_tag = ''
                max_prob = -1
                for tagA in viterbi[i - 1]:
                    # for each tagA
                    # calculate the transition prob P(tagB|tagA)
                    transition_prob = 0
                    if tagB not in transition[tagA]:
                        transition_prob = transition[tagA]['UNK']
                    else:
                        transition_prob = transition[tagA][tagB]

                    if viterbi[i - 1][tagA][0] + transition_prob + emission_prob > max_prob:
                        max_prob = viterbi[i - 1][tagA][0] + transition_prob + emission_prob
                        argmax_tag = tagA
                    
                word_dict[tagB] = (max_prob, argmax_tag)
            viterbi.append(word_dict)

        test_viterbi.append(viterbi)
        print(test_viterbi[-1][-1])
    
    """
    backtracking
    """
    for i in range(len(test_viterbi)):
        # i is the ith sentence in the test set
        return_sentence = []
        sentence_viterbi = test_viterbi[i]
        # for each sentence
        # first find the best tag for the last word
        best_tag = ''
        prev_tag = ''
        max_prob = -1 
        for tag in sentence_viterbi[-1]:
            if sentence_viterbi[-1][tag][0] > max_prob:
                max_prob = sentence_viterbi[-1][tag][0]
                best_tag = tag
                prev_tag = sentence_viterbi[-1][tag][1]
        return_sentence.insert(0, (test[i][-1], best_tag))

        for j in range(len(sentence_viterbi) - 2, -1, -1):
            return_sentence.insert(0, (test[i][j], prev_tag))
            prev_tag = sentence_viterbi[j][prev_tag][1]

        prediction.append(return_sentence)


    print('Your Output is:',prediction,'\n Expected Output is:',output)


if __name__=="__main__":
    main()