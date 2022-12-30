"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math

def dash(word):
        for char in word:
                if char == '-':
                        return True
        return False

def viterbi_3(train, test):
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
    hapax_words_ed = {}
    hapax_words_ly = {}
    hapax_words_er = {}
    hapax_words_s1 = {}
    hapax_words_s2 = {}
    hapax_words_s = {}
    hapax_words_y = {}
    hapax_words_al = {}
    hapax_words_en = {}
    hapax_words_ing = {}
    hapax_words_tic = {}
    hapax_words_ity = {}
    hapax_words_ful = {}
    hapax_words_ous = {}
    hapax_words_acy = {}
    hapax_words_dom = {}
    hapax_words_d = {}
    hapax_words_able = {}
    hapax_words_tion = {}
    hapax_words_sion = {}
    hapax_words_ship = {}
    hapax_words_ment = {}
    hapax_words_ness = {}
    hapax_words_dash = {}
    hapax_words_num = {}
    count_hapax = 0
    count_hapax_ed = 0
    count_hapax_ly = 0
    count_hapax_er = 0
    count_hapax_s1 = 0
    count_hapax_s2 = 0
    count_hapax_s = 0
    count_hapax_y = 0
    count_hapax_al = 0
    count_hapax_en = 0
    count_hapax_ing = 0
    count_hapax_tic = 0
    count_hapax_ity = 0
    count_hapax_ful = 0
    count_hapax_ous = 0
    count_hapax_acy = 0
    count_hapax_dom = 0
    count_hapax_d = 0
    count_hapax_able = 0
    count_hapax_tion = 0
    count_hapax_sion = 0
    count_hapax_ship = 0
    count_hapax_ment = 0
    count_hapax_ness = 0
    count_hapax_dash = 0
    count_hapax_num = 0
    # build the hapax words dictionary
    for tag in emission_dict:
        for word in emission_dict[tag]:
                if emission_dict[tag][word] == 1:
                        if word[-2:] == 'ed':
                                count_hapax_ed += 1
                                if tag not in hapax_words_ed:
                                        hapax_words_ed[tag] = 1
                                else:
                                        hapax_words_ed[tag] += 1
                        if word[-2:] == 'ly':
                                count_hapax_ly += 1
                                if tag not in hapax_words_ly:
                                        hapax_words_ly[tag] = 1
                                else:
                                        hapax_words_ly[tag] += 1
                        if word[-2:] == 'er':
                                count_hapax_er += 1
                                if tag not in hapax_words_er:
                                        hapax_words_er[tag] = 1
                                else:
                                        hapax_words_er[tag] += 1
                        if word[-2:] == "'s":
                                count_hapax_s1 += 1
                                if tag not in hapax_words_s1:
                                        hapax_words_s1[tag] = 1
                                else:
                                        hapax_words_s1[tag] += 1
                        if word[-2:] == "s'":
                                count_hapax_s2 += 1
                                if tag not in hapax_words_s2:
                                        hapax_words_s2[tag] = 1
                                else:
                                        hapax_words_s2[tag] += 1
                        if word[-1] == "s":
                                count_hapax_s += 1
                                if tag not in hapax_words_s:
                                        hapax_words_s[tag] = 1
                                else:
                                        hapax_words_s[tag] += 1
                        if word[-1] == "y":
                                count_hapax_y += 1
                                if tag not in hapax_words_y:
                                        hapax_words_y[tag] = 1
                                else:
                                        hapax_words_y[tag] += 1
                        if word[-2:] == "al":
                                count_hapax_al += 1
                                if tag not in hapax_words_al:
                                        hapax_words_al[tag] = 1
                                else:
                                        hapax_words_al[tag] += 1  
                        if word[-2:] == "en":
                                count_hapax_en += 1
                                if tag not in hapax_words_en:
                                        hapax_words_en[tag] = 1
                                else:
                                        hapax_words_en[tag] += 1  
                        if word[-3:] == "ing":
                                count_hapax_ing += 1
                                if tag not in hapax_words_ing:
                                        hapax_words_ing[tag] = 1
                                else:
                                        hapax_words_ing[tag] += 1  
                        if word[-3:] == "tic":
                                count_hapax_tic += 1
                                if tag not in hapax_words_tic:
                                        hapax_words_tic[tag] = 1
                                else:
                                        hapax_words_tic[tag] += 1 
                        if word[-3:] == "ity":
                                count_hapax_ity += 1
                                if tag not in hapax_words_ity:
                                        hapax_words_ity[tag] = 1
                                else:
                                        hapax_words_ity[tag] += 1 
                        if word[-3:] == "ful":
                                count_hapax_ful += 1
                                if tag not in hapax_words_ful:
                                        hapax_words_ful[tag] = 1
                                else:
                                        hapax_words_ful[tag] += 1 
                        if word[-3:] == "ous":
                                count_hapax_ous += 1
                                if tag not in hapax_words_ous:
                                        hapax_words_ous[tag] = 1
                                else:
                                        hapax_words_ous[tag] += 1 
                        if word[-3:] == "acy":
                                count_hapax_acy += 1
                                if tag not in hapax_words_acy:
                                        hapax_words_acy[tag] = 1
                                else:
                                        hapax_words_acy[tag] += 1 
                        if word[-3:] == "dom":
                                count_hapax_dom += 1
                                if tag not in hapax_words_dom:
                                        hapax_words_dom[tag] = 1
                                else:
                                        hapax_words_dom[tag] += 1 
                        if word[0] == "$":
                                count_hapax_d += 1
                                if tag not in hapax_words_d:
                                        hapax_words_d[tag] = 1
                                else:
                                        hapax_words_d[tag] += 1 
                        if word[-4:] == "able":
                                count_hapax_able += 1
                                if tag not in hapax_words_able:
                                        hapax_words_able[tag] = 1
                                else:
                                        hapax_words_able[tag] += 1   
                        if word[-4:] == "tion":
                                count_hapax_tion += 1
                                if tag not in hapax_words_tion:
                                        hapax_words_tion[tag] = 1
                                else:
                                        hapax_words_tion[tag] += 1  
                        if word[-4:] == "sion":
                                count_hapax_sion += 1
                                if tag not in hapax_words_sion:
                                        hapax_words_sion[tag] = 1
                                else:
                                        hapax_words_sion[tag] += 1
                        if word[-4:] == "ship":
                                count_hapax_ship += 1
                                if tag not in hapax_words_ship:
                                        hapax_words_ship[tag] = 1
                                else:
                                        hapax_words_ship[tag] += 1     
                        if word[-4:] == "ment":
                                count_hapax_ment += 1
                                if tag not in hapax_words_ment:
                                        hapax_words_ment[tag] = 1
                                else:
                                        hapax_words_ment[tag] += 1
                        if word[-4:] == "ness":
                                count_hapax_ness += 1
                                if tag not in hapax_words_ness:
                                        hapax_words_ness[tag] = 1
                                else:
                                        hapax_words_ness[tag] += 1 
                        if dash(word) == True:
                                count_hapax_dash += 1
                                if tag not in hapax_words_dash:
                                        hapax_words_dash[tag] = 1
                                else:
                                        hapax_words_dash[tag] += 1 
                        if word.isnumeric() == True:
                                count_hapax_num += 1
                                if tag not in hapax_words_num:
                                        hapax_words_num[tag] = 1
                                else:
                                        hapax_words_num[tag] += 1  

                        count_hapax += 1
                        if tag not in hapax_words:
                                hapax_words[tag] = 1
                        else:
                                hapax_words[tag] += 1
    
    laplace_hapax = 0.00001
    laplace_hapax_able = 0.01
    # smoothe hapax_words dict
    for tag in emission_dict:
        if tag in hapax_words:
                hapax_words[tag] = (hapax_words[tag] + laplace_hapax) / (count_hapax + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words[tag] = laplace_hapax / (count_hapax + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ed:
                hapax_words_ed[tag] = (hapax_words_ed[tag] + 0.1) / (count_hapax_ed + 0.1 * (len(emission_dict) + 1))
        else:
                hapax_words_ed[tag] = 0.1 / (count_hapax_ed + 0.1 * (len(emission_dict) + 1))
        
        if tag in hapax_words_er:
                hapax_words_er[tag] = (hapax_words_er[tag] + laplace_hapax) / (count_hapax_er + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_er[tag] = laplace_hapax / (count_hapax_er + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ly:
                hapax_words_ly[tag] = (hapax_words_ly[tag] + laplace_hapax) / (count_hapax_ly + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_ly[tag] = laplace_hapax / (count_hapax_ly + laplace_hapax * (len(emission_dict) + 1))

        if tag in hapax_words_s1:
                hapax_words_s1[tag] = (hapax_words_s1[tag] + laplace_hapax) / (count_hapax_s1 + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_s1[tag] = laplace_hapax / (count_hapax_s1 + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_s2:
                hapax_words_s2[tag] = (hapax_words_s2[tag] + laplace_hapax) / (count_hapax_s2 + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_s2[tag] = laplace_hapax / (count_hapax_s2 + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_s:
                hapax_words_s[tag] = (hapax_words_s[tag] + laplace_hapax) / (count_hapax_s + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_s[tag] = laplace_hapax / (count_hapax_s + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_y:
                hapax_words_y[tag] = (hapax_words_y[tag] + laplace_hapax) / (count_hapax_y + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_y[tag] = laplace_hapax / (count_hapax_y + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_al:
                hapax_words_al[tag] = (hapax_words_al[tag] + laplace_hapax) / (count_hapax_al + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_al[tag] = laplace_hapax / (count_hapax_al + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_en:
                hapax_words_en[tag] = (hapax_words_en[tag] + laplace_hapax) / (count_hapax_en + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_en[tag] = laplace_hapax / (count_hapax_en + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ing:
                hapax_words_ing[tag] = (hapax_words_ing[tag] + laplace_hapax) / (count_hapax_ing + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_ing[tag] = laplace_hapax / (count_hapax_ing + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_tic:
                hapax_words_tic[tag] = (hapax_words_tic[tag] + laplace_hapax) / (count_hapax_tic + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_tic[tag] = laplace_hapax / (count_hapax_tic + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ity:
                hapax_words_ity[tag] = (hapax_words_ity[tag] + laplace_hapax) / (count_hapax_ity + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_ity[tag] = laplace_hapax / (count_hapax_ity + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ful:
                hapax_words_ful[tag] = (hapax_words_ful[tag] + laplace_hapax) / (count_hapax_ful + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_ful[tag] = laplace_hapax / (count_hapax_ful + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ous:
                hapax_words_ous[tag] = (hapax_words_ous[tag] + laplace_hapax) / (count_hapax_ous + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_ous[tag] = laplace_hapax / (count_hapax_ous + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_acy:
                hapax_words_acy[tag] = (hapax_words_acy[tag] + laplace_hapax) / (count_hapax_acy + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_acy[tag] = laplace_hapax / (count_hapax_acy + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_dom:
                hapax_words_dom[tag] = (hapax_words_dom[tag] + laplace_hapax) / (count_hapax_dom + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_dom[tag] = laplace_hapax / (count_hapax_dom + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_d:
                hapax_words_d[tag] = (hapax_words_d[tag] + laplace_hapax) / (count_hapax_d + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_d[tag] = laplace_hapax / (count_hapax_d + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_able:
                hapax_words_able[tag] = (hapax_words_able[tag] + laplace_hapax_able) / (count_hapax_able + laplace_hapax_able * (len(emission_dict) + 1))
        else:
                hapax_words_able[tag] = laplace_hapax_able / (count_hapax_able + laplace_hapax_able * (len(emission_dict) + 1))
        
        if tag in hapax_words_tion:
                hapax_words_tion[tag] = (hapax_words_tion[tag] + laplace_hapax) / (count_hapax_tion + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_tion[tag] = laplace_hapax / (count_hapax_tion + laplace_hapax * (len(emission_dict) + 1))

        if tag in hapax_words_sion:
                hapax_words_sion[tag] = (hapax_words_sion[tag] + laplace_hapax) / (count_hapax_sion + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_sion[tag] = laplace_hapax / (count_hapax_sion + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ship:
                hapax_words_ship[tag] = (hapax_words_ship[tag] + laplace_hapax) / (count_hapax_ship + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_ship[tag] = laplace_hapax / (count_hapax_ship + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ment:
                hapax_words_ment[tag] = (hapax_words_ment[tag] + laplace_hapax) / (count_hapax_ment + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_ment[tag] = laplace_hapax / (count_hapax_ment + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_ness:
                hapax_words_ness[tag] = (hapax_words_ness[tag] + laplace_hapax) / (count_hapax_ness + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_ness[tag] = laplace_hapax / (count_hapax_ness + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_dash:
                hapax_words_dash[tag] = (hapax_words_dash[tag] + laplace_hapax) / (count_hapax_dash + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_dash[tag] = laplace_hapax / (count_hapax_dash + laplace_hapax * (len(emission_dict) + 1))
        
        if tag in hapax_words_num:
                hapax_words_num[tag] = (hapax_words_num[tag] + laplace_hapax) / (count_hapax_num + laplace_hapax * (len(emission_dict) + 1))
        else:
                hapax_words_num[tag] = laplace_hapax / (count_hapax_num + laplace_hapax * (len(emission_dict) + 1))



    laplace_em = 0.00001
    # compute emission probability
    for tag in emission_dict:
        V = len(emission_dict[tag])
        n = sum(emission_dict[tag].values())
        for word in emission_dict[tag]:
            emission_dict[tag][word] = math.log((emission_dict[tag][word] + laplace_em * hapax_words[tag]) / (n + laplace_em * hapax_words[tag] * (V + 1)))
        # assign probability for unknown words
        emission_dict[tag]['UNK'] = math.log(laplace_em * hapax_words[tag]/ (n + laplace_em * hapax_words[tag] * (V + 1)))
        emission_dict[tag]['ED'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ed[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ed[tag] * (V + 1)))
        emission_dict[tag]['LY'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ly[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ly[tag] * (V + 1)))
        emission_dict[tag]['ER'] = math.log(laplace_em * hapax_words[tag] * hapax_words_er[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_er[tag] * (V + 1)))
        emission_dict[tag]['S1'] = math.log(laplace_em * hapax_words[tag] * hapax_words_s1[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_s1[tag] * (V + 1)))
        emission_dict[tag]['S2'] = math.log(laplace_em * hapax_words[tag] * hapax_words_s2[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_s2[tag] * (V + 1)))
        emission_dict[tag]['S'] = math.log(laplace_em * hapax_words[tag] * hapax_words_s[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_s[tag] * (V + 1)))
        emission_dict[tag]['Y'] = math.log(laplace_em * hapax_words[tag] * hapax_words_y[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_y[tag] * (V + 1)))
        emission_dict[tag]['AL'] = math.log(laplace_em * hapax_words[tag] * hapax_words_al[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_al[tag] * (V + 1)))
        emission_dict[tag]['EN'] = math.log(laplace_em * hapax_words[tag] * hapax_words_en[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_en[tag] * (V + 1)))
        emission_dict[tag]['ING'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ing[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ing[tag] * (V + 1)))
        emission_dict[tag]['TIC'] = math.log(laplace_em * hapax_words[tag] * hapax_words_tic[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_tic[tag] * (V + 1)))
        emission_dict[tag]['ITY'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ity[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ity[tag] * (V + 1)))
        emission_dict[tag]['FUL'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ful[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ful[tag] * (V + 1)))
        emission_dict[tag]['OUS'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ous[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ous[tag] * (V + 1)))
        emission_dict[tag]['ACY'] = math.log(laplace_em * hapax_words[tag] * hapax_words_acy[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_acy[tag] * (V + 1)))
        emission_dict[tag]['DOM'] = math.log(laplace_em * hapax_words[tag] * hapax_words_dom[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_dom[tag] * (V + 1)))
        emission_dict[tag]['D'] = math.log(laplace_em * hapax_words[tag] * hapax_words_d[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_d[tag] * (V + 1)))
        emission_dict[tag]['ABLE'] = math.log(laplace_em * hapax_words[tag] * hapax_words_able[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_able[tag] * (V + 1)))
        emission_dict[tag]['TION'] = math.log(laplace_em * hapax_words[tag] * hapax_words_tion[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_tion[tag] * (V + 1)))
        emission_dict[tag]['SION'] = math.log(laplace_em * hapax_words[tag] * hapax_words_sion[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_sion[tag] * (V + 1)))
        emission_dict[tag]['SHIP'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ship[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ship[tag] * (V + 1)))
        emission_dict[tag]['MENT'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ment[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ment[tag] * (V + 1)))
        emission_dict[tag]['NESS'] = math.log(laplace_em * hapax_words[tag] * hapax_words_ness[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_ness[tag] * (V + 1)))
        emission_dict[tag]['DASH'] = math.log(laplace_em * hapax_words[tag] * hapax_words_dash[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_dash[tag] * (V + 1)))
        emission_dict[tag]['NUM'] = math.log(laplace_em * hapax_words[tag] * hapax_words_num[tag] / (n + laplace_em * hapax_words[tag] * hapax_words_num[tag] * (V + 1)))

    

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
                if first_word[-2:] == 'ed':
                        emission_prob = emission_dict[tag]['ED']
                elif first_word[-2:] == 'er':
                        emission_prob = emission_dict[tag]['ER']
                elif first_word[-2:] == 'ly':
                        emission_prob = emission_dict[tag]['LY']
                elif first_word[-2:] == "'s":
                        emission_prob = emission_dict[tag]['S1']
                elif first_word[-2:] == "s'":
                        emission_prob = emission_dict[tag]['S2']
                elif first_word[-1] == "s":
                        emission_prob = emission_dict[tag]['S']
                elif first_word[-2:] == "al":
                        emission_prob = emission_dict[tag]['AL']
                elif first_word[-2:] == "en":
                        emission_prob = emission_dict[tag]['EN']
                elif first_word[-3:] == "ing":
                        emission_prob = emission_dict[tag]['ING']
                elif first_word[-3:] == "tic":
                        emission_prob = emission_dict[tag]['TIC']
                elif first_word[-3:] == "ity":
                        emission_prob = emission_dict[tag]['ITY']
                elif first_word[-3:] == "ful":
                        emission_prob = emission_dict[tag]['FUL']
                elif first_word[-3:] == "ous":
                        emission_prob = emission_dict[tag]['OUS']
                elif first_word[-3:] == "acy":
                        emission_prob = emission_dict[tag]['ACY']
                elif first_word[-3:] == "dom":
                        emission_prob = emission_dict[tag]['DOM']
                elif first_word[0] == "d":
                        emission_prob = emission_dict[tag]['D']
                # elif first_word[-1] == "y":
                        # emission_prob = emission_dict[tag]['Y']
                elif first_word[-4:] == "able":
                        emission_prob = emission_dict[tag]['ABLE']
                elif first_word[-4:] == "tion":
                        emission_prob = emission_dict[tag]['TION']
                elif first_word[-4:] == "sion":
                        emission_prob = emission_dict[tag]['SION']
                elif first_word[-4:] == "ship":
                        emission_prob = emission_dict[tag]['SHIP']
                elif first_word[-4:] == "ment":
                        emission_prob = emission_dict[tag]['MENT']
                elif first_word[-4:] == "ness":
                        emission_prob = emission_dict[tag]['NESS']
                elif dash(first_word) == True:
                        emission_prob = emission_dict[tag]['DASH']
                elif first_word.isnumeric() == True:
                        emission_prob = emission_dict[tag]['NUM']
                else:
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
                     '''
                     if ((word[-2:] == 'ed' or word[-2:] == 'er' or word[-2:] == 'ly' or word[-2:] == "'s" or word[-2:] == "s'" or word[-1] == "s"
                     or word[-2:] == "al" or word[-3:] == "ing" or word[-3:] == "tic" or word[-3:] == "ity" or word[-3:] == "ful" or word[-3:] == "ous" or 
                     word[-4:] == "able" or word[-4:] == "tion" or word[-4:] == "sion" or word[-4:] == "ship" or word[-4:] == "ment" or word[-4:] == "ness"
                     or word.isnumeric() == True) == False):
                        print(word)
                     '''

                     if word[-2:] == 'ed':
                           emission_prob = emission_dict[tagB]['ED']
                     elif word[-2:] == 'er':
                           emission_prob = emission_dict[tagB]['ER']
                     elif word[-2:] == 'ly':
                           emission_prob = emission_dict[tagB]['LY']
                     elif word[-2:] == "'s":
                           emission_prob = emission_dict[tagB]['S1']
                     elif word[-2:] == "s'":
                           emission_prob = emission_dict[tagB]['S2']
                     elif word[-1] == "s":
                           emission_prob = emission_dict[tagB]['S']
                     elif word[-2:] == "al":
                           emission_prob = emission_dict[tagB]['AL']
                     elif word[-2:] == "en":
                           emission_prob = emission_dict[tagB]['EN']
                     elif word[-3:] == "ing":
                           emission_prob = emission_dict[tagB]['ING']
                     elif word[-3:] == "tic":
                           emission_prob = emission_dict[tagB]['TIC']
                     elif word[-3:] == "ity":
                           emission_prob = emission_dict[tagB]['ITY']
                     elif word[-3:] == "ful":
                           emission_prob = emission_dict[tagB]['FUL']
                     elif word[-3:] == "ous":
                           emission_prob = emission_dict[tagB]['OUS']
                     elif word[-3:] == "acy":
                           emission_prob = emission_dict[tagB]['ACY']
                     elif word[-3:] == "dom":
                           emission_prob = emission_dict[tagB]['DOM']
                     elif word[0] == "d":
                           emission_prob = emission_dict[tagB]['D']
                     # elif word[-1] == "y":
                           # emission_prob = emission_dict[tagB]['Y']
                     elif word[-4:] == "able":
                           emission_prob = emission_dict[tagB]['ABLE']
                     elif word[-4:] == "tion":
                           emission_prob = emission_dict[tagB]['TION']
                     elif word[-4:] == "sion":
                           emission_prob = emission_dict[tagB]['SION']
                     elif word[-4:] == "ship":
                           emission_prob = emission_dict[tagB]['SHIP']
                     elif word[-4:] == "ment":
                           emission_prob = emission_dict[tagB]['MENT']
                     elif word[-4:] == "ness":
                           emission_prob = emission_dict[tagB]['NESS']
                     elif dash(word) == True:
                           emission_prob = emission_dict[tagB]['DASH']
                     elif word.isnumeric() == True:
                           emission_prob = emission_dict[tagB]['NUM']
                     else:
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