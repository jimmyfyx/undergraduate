"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""
import math

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # initialize two embedded dict for emission and transition probability
    emission_dict = {}
    transition_dict = {}

    # iterate through the training data
    for sentence in train:
        for i in range(len(sentence)):
            word = sentence[i][0]
            tag = sentence[i][1]

            # modify emission_dict
            if i < len(sentence) - 1 and i > 0:
                if tag not in emission_dict:
                    emission_dict[tag] = {word:1}
                else:
                    if word not in emission_dict[tag]:
                        emission_dict[tag][word] = 1
                    else:
                        emission_dict[tag][word] += 1
            
            # modify transition_dict
            if i < len(sentence) - 1:
                next_tag = sentence[i + 1][1]
                if tag not in transition_dict:
                    transition_dict[tag] = {next_tag:1}
                else:
                    if next_tag not in transition_dict[tag]:
                        transition_dict[tag][next_tag] = 1
                    else:
                        transition_dict[tag][next_tag] += 1
    
    hapax_words = {}
    count_hapax = 0
    # build the hapax words dictionary
    for tag in emission_dict:
        for word in emission_dict[tag]:
                if emission_dict[tag][word] == 1:
                        count_hapax += 1
                        if tag not in hapax_words:
                                hapax_words[tag] = 1
                        else:
                                hapax_words[tag] += 1
    
    laplace_hapax = 0.00001
    # smoothe hapax_words dict
    for tag in emission_dict:
        if tag in hapax_words:
                hapax_words[tag] = (hapax_words[tag] + laplace_hapax) / (count_hapax + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words[tag] = laplace_hapax / (count_hapax + laplace_hapax * (len(emission_dict) + 1))

    laplace_em = 0.00001
    # compute emission probability
    for tag in emission_dict:
        V = len(emission_dict[tag])
        n = sum(emission_dict[tag].values())
        for word in emission_dict[tag]:
            emission_dict[tag][word] = math.log((emission_dict[tag][word] + laplace_em * hapax_words[tag]) / (n + laplace_em * hapax_words[tag] * (V + 1)))
        # assign probability for unknown words
        emission_dict[tag]['UNK'] = math.log(laplace_em * hapax_words[tag]/ (n + laplace_em * hapax_words[tag] * (V + 1)))
    

    laplace_tr = 0.00001
    # compute transition probability
    for tag in transition_dict:
        V = len(transition_dict[tag])
        n = sum(transition_dict[tag].values())
        for next_tag in transition_dict[tag]:
            transition_dict[tag][next_tag] = (transition_dict[tag][next_tag] + laplace_tr) / (n + laplace_tr * (V + 1))
            transition_dict[tag][next_tag] = math.log(transition_dict[tag][next_tag])
        # assign probability for unseen following tags
        transition_dict[tag]['UNK'] = math.log(laplace_tr / (n + laplace_tr * (V + 1)))


    """
    implement viterbi algorithm
    """
    test_viterbi = []
    # the first iteration through the test set is used to construct the viterbi structure
    for sentence in test:
        # each sentence owns a viterbi 
        viterbi = []
        empty = {}
        viterbi.append(empty)

        # initialize the first column
        first_word = sentence[1]
        first_dict = {}
        # iterate through all the possible tags
        for tag in emission_dict:
            # calculate transition prob
            transition_prob = 0
            if tag not in transition_dict['START']:
                transition_prob = transition_dict['START']['UNK']
            else:
                transition_prob = transition_dict['START'][tag]
            # calculate emission prob
            emission_prob = 0
            if first_word not in emission_dict[tag]:
                emission_prob = emission_dict[tag]['UNK']
            else:
                emission_prob = emission_dict[tag][first_word]

            first_dict[tag] = (transition_prob + emission_prob, 'START')
        viterbi.append(first_dict)

        # iterate through the rest of the sentence
        for i in range(2, len(sentence), 1):
            if sentence[i] == 'END':
                break
            word = sentence[i]
            word_dict = {}
            # for each word iterate through all the possible tags (tagB)
            for tagB in emission_dict:
                # for each tagB iterate through all the possible previous tags
                # calculate emission prob
                emission_prob = 0
                if word not in emission_dict[tagB]:
                    emission_prob = emission_dict[tagB]['UNK']
                else:
                    emission_prob = emission_dict[tagB][word]

                argmax_tag = ''
                max_prob = -float('inf')
                for tagA in viterbi[i - 1]:
                    # for each tagA
                    # calculate the transition prob P(tagB|tagA)
                    transition_prob = 0
                    if tagB not in transition_dict[tagA]:
                        transition_prob = transition_dict[tagA]['UNK']
                    else:
                        transition_prob = transition_dict[tagA][tagB]

                    if viterbi[i - 1][tagA][0] + transition_prob + emission_prob > max_prob:
                        max_prob = viterbi[i - 1][tagA][0] + transition_prob + emission_prob
                        argmax_tag = tagA  
                word_dict[tagB] = (max_prob, argmax_tag)
            viterbi.append(word_dict)
        viterbi.append(empty)
        test_viterbi.append(viterbi)
    

    """
    backtracking
    """
    return_list = []
    for i in range(len(test_viterbi)):
        # i is the ith sentence in the test set
        return_sentence = []
        return_sentence.append(('END', 'END'))
        sentence_viterbi = test_viterbi[i]
        # for each sentence
        # first find the best tag for the last word
        best_tag = ''
        prev_tag = ''
        max_prob = -float('inf')   
        for tag in sentence_viterbi[-2]:
            if sentence_viterbi[-2][tag][0] > max_prob:
                max_prob = sentence_viterbi[-2][tag][0]
                best_tag = tag
                prev_tag = sentence_viterbi[-2][tag][1]
        return_sentence.insert(0, (test[i][-2], best_tag))

        for j in range(len(sentence_viterbi) - 3, -1, -1):
            if j == 0:
                break
            return_sentence.insert(0, (test[i][j], prev_tag))
            prev_tag = sentence_viterbi[j][prev_tag][1]
        return_sentence.insert(0, ('START', 'START'))

        return_list.append(return_sentence)

    return return_list