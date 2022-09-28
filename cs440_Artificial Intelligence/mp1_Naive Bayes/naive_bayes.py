# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.05, pos_prior=0.8,silently=False):
    print_paramter_vals(laplace,pos_prior)

    """
    Phase_1: Process training set
    """
    # initialize parameters
    yhats = []
    num_reviews = len(train_set)
    pos_dict = {}
    neg_dict = {}
    pos_prob_dict = {}   # store P{w|positive} for each word type
    neg_prob_dict = {}   # store P{W|negative} for each word type
    pos_words = 0
    neg_words= 0
    
    # loop through the training set
    for i in range(num_reviews):
        # loop through each word in each review
        for j in range(len(train_set[i])):
            if train_labels[i] == 1:
                pos_words += 1
                if train_set[i][j] in pos_dict.keys():
                    pos_dict[train_set[i][j]] += 1
                else:
                    pos_dict[train_set[i][j]] = 1
            else:
                neg_words += 1
                if train_set[i][j] in neg_dict.keys():
                    neg_dict[train_set[i][j]] += 1
                else:
                    neg_dict[train_set[i][j]] = 1

    for pos_word in pos_dict.keys():
        pos_prob_dict[pos_word] = (pos_dict[pos_word] + laplace) / (pos_words + laplace * (len(pos_dict) + 1))
    for neg_word in neg_dict.keys():
        neg_prob_dict[neg_word] = (neg_dict[neg_word] + laplace) / (neg_words + laplace * (len(neg_dict) + 1))
    pos_prob_dict["UNK"] = laplace / (pos_words + laplace * (len(pos_dict) + 1))
    neg_prob_dict["UNK"] = laplace / (neg_words + laplace * (len(neg_dict) + 1))
    
    """
    Phase_2: Development Phase
    """
    num_reviews_dev = len(dev_set)
    # loop through the development set
    for i in range(num_reviews_dev):
        neg_prob = math.log2(1 - pos_prior)
        pos_prob = math.log2(pos_prior)
        for j in range(len(dev_set[i])):
            if dev_set[i][j] in pos_prob_dict.keys():
                pos_prob += math.log2(pos_prob_dict[dev_set[i][j]])
            else:
                pos_prob += math.log2(pos_prob_dict["UNK"])
            
            if dev_set[i][j] in neg_prob_dict.keys():
                neg_prob += math.log2(neg_prob_dict[dev_set[i][j]])
            else:
                neg_prob += math.log2(neg_prob_dict["UNK"])

        # assign label for each review
        if pos_prob > neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats



def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.1, bigram_laplace=0.0001, bigram_lambda=0.09,pos_prior=0.8, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # initializations
    yhats = []
    num_reviews = len(train_set)
    pos_dict = {}
    neg_dict = {}
    pos_dict_bi = {}
    neg_dict_bi = {}
    pos_prob_dict = {}   # store P{w|positive} for each word type (unigram)
    neg_prob_dict = {}   # store P{W|negative} for each word type (unigram)
    pos_prob_dict_bi = {}
    neg_prob_dict_bi = {}
    pos_words = 0        # the total number of words in positive reviews
    neg_words= 0         # the total number of words in negative reviews
    pos_words_pair = 0
    neg_words_pair = 0

    # loop through the training set
    for i in range(num_reviews):
        # loop through each word in each review
        for j in range(len(train_set[i])):
            if train_labels[i] == 1:
                # collect data for the unigram model
                pos_words += 1
                if train_set[i][j] in pos_dict.keys():
                    pos_dict[train_set[i][j]] += 1
                else:
                    pos_dict[train_set[i][j]] = 1
                
                # collect data for the bigram model
                if j + 1 < len(train_set[i]):
                    pos_words_pair += 1
                    pair = train_set[i][j] + train_set[i][j + 1]
                    if pair in pos_dict_bi.keys():
                        pos_dict_bi[pair] += 1
                    else:
                        pos_dict_bi[pair] = 1
            else:
                # collect data for the unigram model
                neg_words += 1
                if train_set[i][j] in neg_dict.keys():
                    neg_dict[train_set[i][j]] += 1
                else:
                    neg_dict[train_set[i][j]] = 1
                
                # collect data for the bigram model
                if j + 1 < len(train_set[i]):
                    neg_words_pair += 1
                    pair = train_set[i][j] + train_set[i][j + 1]
                    if pair in neg_dict_bi.keys():
                        neg_dict_bi[pair] += 1
                    else:
                        neg_dict_bi[pair] = 1

    # probability for unigram model
    for pos_word in pos_dict.keys():
        pos_prob_dict[pos_word] = (pos_dict[pos_word] + unigram_laplace) / (pos_words + unigram_laplace * (len(pos_dict) + 1))
    for neg_word in neg_dict.keys():
        neg_prob_dict[neg_word] = (neg_dict[neg_word] + unigram_laplace) / (neg_words + unigram_laplace * (len(neg_dict) + 1))
    
    # probability for bigram model
    for pos_word_pair in pos_dict_bi.keys():
        pos_prob_dict_bi[pos_word_pair] = (pos_dict_bi[pos_word_pair] + bigram_laplace) / (pos_words_pair + bigram_laplace * (len(pos_dict_bi) + 1))
    for neg_word_pair in neg_dict_bi.keys():
        neg_prob_dict_bi[neg_word_pair] = (neg_dict_bi[neg_word_pair] + bigram_laplace) / (neg_words_pair + bigram_laplace * (len(neg_dict_bi) + 1))

    pos_prob_dict["UNK"] = unigram_laplace / (pos_words + unigram_laplace * (len(pos_dict) + 1))
    neg_prob_dict["UNK"] = unigram_laplace / (neg_words + unigram_laplace * (len(neg_dict) + 1))
    pos_prob_dict_bi["UNK"] = bigram_laplace / (pos_words_pair + bigram_laplace *(len(pos_dict_bi) + 1))
    neg_prob_dict_bi["UNK"] = bigram_laplace / (neg_words_pair + bigram_laplace *(len(neg_dict_bi) + 1))


    num_reviews_dev = len(dev_set)
    # loop through the development set
    for i in range(num_reviews_dev):
        neg_prob_uni = math.log2(1 - pos_prior)
        pos_prob_uni = math.log2(pos_prior)
        neg_prob_bi = math.log2(1 - pos_prior)
        pos_prob_bi = math.log2(pos_prior)
        for j in range(len(dev_set[i])):
            if dev_set[i][j] in pos_prob_dict.keys():
                pos_prob_uni += math.log2(pos_prob_dict[dev_set[i][j]])
            else:
                pos_prob_uni += math.log2(pos_prob_dict["UNK"])
            
            if dev_set[i][j] in neg_prob_dict.keys():
                neg_prob_uni += math.log2(neg_prob_dict[dev_set[i][j]])
            else:
                neg_prob_uni += math.log2(neg_prob_dict["UNK"])
            
            if j + 1 < len(dev_set[i]):
                pair = dev_set[i][j] + dev_set[i][j + 1]
                if pair in pos_prob_dict_bi.keys():
                    pos_prob_bi += math.log2(pos_prob_dict_bi[pair])
                else:
                    pos_prob_bi += math.log2(pos_prob_dict_bi["UNK"])
                
                if pair in neg_prob_dict_bi.keys():
                    neg_prob_bi += math.log2(neg_prob_dict_bi[pair])
                else:
                    neg_prob_bi += math.log2(neg_prob_dict_bi["UNK"])

        final_prob_pos = (1 - bigram_lambda) * pos_prob_uni + bigram_lambda * pos_prob_bi
        final_prob_neg = (1 - bigram_lambda) * neg_prob_uni + bigram_lambda * neg_prob_bi

        # assign label for each review
        if final_prob_pos > final_prob_neg:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats

    """
    yhats = []
    for i in range(len(dev_set)):
        yhats.append(-1)
    return yhats
    """

