"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
	
    tag_dict = {}
    words_dict = {}
    # for each word, use a embedded dictionary to record the tag count for each word
    for sentence in train: 
        for word_tag in sentence:
            # add tag to tag_dict
            if word_tag[1] not in tag_dict:
                tag_dict[word_tag[1]] = 1
            else:
                tag_dict[word_tag[1]] += 1

            if word_tag[0] not in words_dict:
                words_dict[word_tag[0]] = {word_tag[1]:1}
            else:
                if word_tag[1] not in words_dict[word_tag[0]]:
                    words_dict[word_tag[0]][word_tag[1]] = 1
                else:
                    words_dict[word_tag[0]][word_tag[1]] += 1 


    return_list = []
    for sentence in test:
        sentence_list = []
        for word in sentence:
            if word not in words_dict:
                sentence_list.append((word, max(tag_dict, key=tag_dict.get)))
            else:
                assign_tag = max(words_dict[word], key=words_dict[word].get)
                sentence_list.append((word, assign_tag))
        return_list.append(sentence_list)
    
    return return_list