#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 08/10/2021 16:00
# @Author  : Shuyin Ouyang
# @File    : assignment.py


import re
import random
import traceback
import time
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


# 4.3.1 Preprocessing each line (10 marks)
# input : a string
def preprocess_line(line):
    # define a pattern that can match the characters expect a-z, A-Z, 0-9, dot, whitespace
    cop = re.compile("[^a-z^A-Z^0-9^.^ ]")
    # define a pattern that can match the characters within 0-9
    num = re.compile("[0-9]")
    # remove non-compliant characters
    line = cop.sub('', line)
    # convert all digits to ‘0’
    line = num.sub('0', line)
    # lowercase all remaining characters
    line = line.lower()
    # return the preprocessed line
    return line


def preprocess_line_new(line):
    # define a trainslation dic
    trans_dic_de = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
    trans_dic_es = {'á': 'a', 'é': 'e', 'ú': 'u', 'ñ': 'n', 'í': 'í'}
    for char in trans_dic_de:
        line = line.replace(char, trans_dic_de[char])
    for char in trans_dic_es:
        line = line.replace(char, trans_dic_es[char])
    # define a pattern that can match the characters expect a-z, A-Z, 0-9, dot, whitespace
    cop = re.compile("[^a-z^A-Z^0-9^.^ ]")
    # define a pattern that can match the characters within 0-9
    num = re.compile("[0-9]")
    # remove non-compliant characters
    line = cop.sub('', line)
    # convert all digits to ‘0’
    line = num.sub('0', line)
    # lowercase all remaining characters
    line = line.lower()
    # return the preprocessed line
    return line


# 4.3.2 Examining a pre-trained model (10 marks)
# Do not need coding

# add_alpha function
# input: trigram-count dictionary, bigram-count dictionary, alpha with default value 0.01
def add_alpha(tri_count_dic, bi_count_dic, alpha=0.01):
    # define tri_prob_dic to store the trigram and its probability(key: characters, value: probability)
    tri_prob_dic = defaultdict(float)
    # read given model-br.en for setting the keys in tri_prob_dic
    with open('data/model-br.en', 'r') as f:
        for line in f:
            # for every line in the file, split the line with '\t', then generate a list [trigram, probability]
            content = line.split('\t')
            tri_prob_dic[content[0]] = 0

    # total number of tri-characters
    tri_num = len(tri_count_dic)
    for tri in tri_prob_dic:
        # for each trigram in the tri_prob_dic, calculate its probability with add-alpha method
        try:
            tri_prob_dic[tri] = (tri_count_dic[tri] + alpha) / (bi_count_dic[tri[:-1]] + alpha * tri_num)
        # if the denominator is so small that the computer regards it as 0, we return that trigram probability as 0
        except ZeroDivisionError:
            tri_prob_dic[tri] = 0
            continue
    return tri_prob_dic


# 4.3.3 Implementing a model: description and example probabilities (35 marks)
# input: the location of train_file and output_file, and the value of alpha
def model_implement(train_file, output_file, alpha):
    # create a dictionary for data restore
    # key: characters, value: count
    tri_count_dic = defaultdict(int)
    bi_count_dic = defaultdict(int)

    # read training file
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            # preprocess the line
            line_processed = preprocess_line_new(line)
            # add '##' at the begin and '#' at the end of the sequence
            line_processed = '##' + line_processed + '#'
            # *collect counts for both trigram and bigram
            for i in range(len(line_processed) - 1):
                bi_count_dic[line_processed[i:i + 2]] += 1
                if i + 3 <= len(line_processed):
                    tri_count_dic[line_processed[i:i + 3]] += 1
    # estimate probabilities
    # use add-alpha method to calculate the probability of each trigram
    tri_prob_dic = add_alpha(tri_count_dic, bi_count_dic, alpha)

    # write the probabilities into a file
    try:
        with open(output_file, 'w') as ff:
            for tri in tri_prob_dic:
                ff.write('%s\t%.3e\n' % (tri, tri_prob_dic[tri]))
    # If the code doesn't work, print the reason of it
    except Exception as e:
        print(e.with_traceback())


# using dev set to optimize a better alpha in given alpha list
def train_LM():
    # key: alpha value:[MSE, perplexity, running time]
    LM_dic = {}
    # create a dictionary for data restore
    # key: characters, value: count
    tri_count_dic = defaultdict(int)
    bi_count_dic = defaultdict(int)

    # read training file (take the English one as an example)
    with open('data/training.en', 'r') as f:
        for line in f:
            # preprocess the line
            line_processed = preprocess_line(line)
            # add '##' at the begin and end of the sequence
            line_processed = '##' + line_processed + '#'
            # collect counts for both trigram and bigram
            for i in range(len(line_processed) - 1):
                bi_count_dic[line_processed[i:i + 2]] += 1
                if i + 3 <= len(line_processed):
                    tri_count_dic[line_processed[i:i + 3]] += 1
        # read dev file
        # add '##' at the beginning and '#' at the end of the sequence
        dev_sequence = '##'
        with open('data/dev', 'r', encoding='utf-8') as f:
            for line in f:
                # preprocess the line
                line_process = preprocess_line(line)
                # directly add each line into the dev_sequence without adding '#' between each line
                dev_sequence += line_process
        # finally return only one sequence of dev file
        dev_sequence += '#'

    # define a benchmark for different add-alpha method's alpha value
    alpha_0_dic = defaultdict(float)
    # alpha_list = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alpha_list = [0, 0.01]
    alpha_list += [(0.1) * i for i in range(1, 10)]
    best_alpha = 0
    min_perplexity = -1
    # for each alpha in alpha list we train a model and record its MSE, perplexity and running time
    for alpha in alpha_list:
        try:
            # record the start time
            begin_time = time.time()
            # add-alpha funtion
            add_alpha_tri_dic = add_alpha(tri_count_dic, bi_count_dic, alpha)
            mse = 0
            perplexity = 0
            # when alpha = 0, we set MSE = -1, which is different from others
            if alpha == 0:
                # restore the benchmark into alpha_0_dic
                alpha_0_dic = add_alpha_tri_dic
                # record the end time
                end_time = time.time()
                # restore the [MSE, perplexity and running time] into LM_dic
                LM_dic[alpha] = [-1, -1, end_time - begin_time]
                continue
            else:
                # compute MSE
                for tri in add_alpha_tri_dic.keys():
                    mse += (add_alpha_tri_dic[tri] - alpha_0_dic[tri]) ** 2
                # compute complexity
                for i in range(2, len(dev_sequence)):
                    perplexity += -np.log2(add_alpha_tri_dic[dev_sequence[i - 2: i + 1]])
                perplexity = 2 ** (perplexity * (1 / (len(dev_sequence) - 2)))
                # record the end time
                end_time = time.time()
                # restore the [MSE, perplexity and running time] into LM_dic
                LM_dic[alpha] = [mse / len(add_alpha_tri_dic), perplexity, end_time - begin_time]
                if min_perplexity == -1:
                    min_perplexity = perplexity
                    best_alpha = alpha
                else:
                    if perplexity < min_perplexity:
                        min_perplexity = perplexity
                        best_alpha = alpha
        # If code doesn't work, print the current loop's alpha value and its reason
        except Exception:
            print('alpha = %.3e' % (alpha))
            print(traceback.format_exc())
    # clear the plot container
    plt.figure()
    # plot the line of alpha-perplexity
    plt.plot([i for i in alpha_list[1:]], [LM_dic[alpha][1] for alpha in alpha_list[1:]], \
             c='red', markerfacecolor='blue', marker='o')
    # print the value of the points on line
    for a, b in zip([i for i in alpha_list[1:]], [LM_dic[alpha][1] for alpha in alpha_list[1:]]):
        plt.text(round(a, 2), round(b, 5), (round(a, 2), round(b, 5)), fontsize=10)
    # show the plot
    plt.title('The plot of alpha-perplexity')
    plt.ylabel("perplexity")
    plt.xlabel("alpha")

    plt.show()
    # return the LM_dic and best alpha
    return LM_dic, best_alpha


# 4.3.4 Generating from models (15 marks)
# input: the location of training model
def generate_from_LM(model):
    # target sequence
    generated_sequence = '##'
    # load the trigram-probability dictionary from local file
    tri_prob_dic = defaultdict(float)
    with open(model, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split('\t')
            tri_prob_dic[content[0]] = float(content[1])
    for _ in range(300):
        try:
            # if the length of sequence bigger than 302, break the loop
            if len(generated_sequence) >= 302:
                break
            # If face the situation that only the last one character is '#'
            if generated_sequence[-2:] != '##' and generated_sequence[-1] == '#':
                # generate another '#' for further sequence successfully generation
                generated_sequence += '#'
                continue
            # get the last two characters in the current generated_sequence
            cur_prior_characters = generated_sequence[len(generated_sequence) - 2:]
            # find the trigrams start with those two characters
            characters_chosen_list = [tri for tri in list(tri_prob_dic.keys()) \
                                      if tri.startswith(cur_prior_characters)]
            # find the trigrams' corresponding probability
            characters_chosen_prob = [tri_prob_dic[tri] for tri in characters_chosen_list]
            # random generate the next character depends on the probabilities
            generate_character = random.choices(characters_chosen_list, characters_chosen_prob)[0]
            # add the generated character into the generated sequence
            generated_sequence += generate_character[-1]
        # If the code doesn't work, print the reason and jump out from the current loop to continue
        except Exception:
            print(traceback.format_exc())
            continue
    # return 300 characters in the sequence expect the first '##'
    return generated_sequence[:302] + '#'


# 4.3.5 Computing perplexity (15 marks)
# input: the location of test_file and model
def compute_perplexity(test_file, model):
    # preprocess the test file
    test_sequence = '##'
    # read the test sequence from the test file
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_process = preprocess_line_new(line)
            test_sequence += line_process
    test_sequence += '#'
    # load the dictionary of trigram-prob.
    tri_prob_dic = defaultdict(float)
    # read the our model from the local file
    with open(model, 'r') as f:
        for line in f:
            content = line.split('\t')
            tri_prob_dic[content[0]] = float(content[1])

    # compute the perplexity of test sequence
    perplexity = 0
    for i in range(2, len(test_sequence)):
        perplexity += -np.log2(tri_prob_dic[test_sequence[i - 2: i + 1]])
    perplexity = 2 ** (perplexity * (1 / (len(test_sequence) - 2)))
    # return perplexity
    return perplexity

# 4.3.6 Extra (15 marks)
def extra():
    # optimize the proper alpha
    best_alpha = 0.01
    # define the train_file_set
    train_file_list = ['data/training.en', 'data/training.de', 'data/training.es']
    # implement the three model with best alpha
    for train_file in train_file_list:
        output_file = 'train_res1/model.' + train_file.split('.')[1]
        model_implement(train_file, output_file, best_alpha)
    perplexity_list = []
    # calculate three model's perplexity to the test file
    for train_file in train_file_list:
        model = 'train_res1/model.' + train_file.split('.')[1]
        perplexity = compute_perplexity('data/test', model)
        perplexity_list.append(perplexity)
    print("Test file: English")
    print('Perplexity:\nmodel.en: %s, model.de: %s, model.es: %s' % \
          (perplexity_list[0], perplexity_list[1], perplexity_list[2]))
    perplexity_list = []
    for train_file in train_file_list:
        model = 'train_res1/model.' + train_file.split('.')[1]
        perplexity = compute_perplexity('data/test_spanish.txt', model)
        perplexity_list.append(perplexity)
    print("Test file: Spanish")
    print('Perplexity:\nmodel.en: %s, model.de: %s, model.es: %s' % \
          (perplexity_list[0], perplexity_list[1], perplexity_list[2]))
    perplexity_list = []
    for train_file in train_file_list:
        model = 'train_res1/model.' + train_file.split('.')[1]
        perplexity = compute_perplexity('data/test_german.txt', model)
        perplexity_list.append(perplexity)
    print("Test file: German")
    print('Perplexity:\nmodel.en: %s, model.de: %s, model.es: %s' % \
          (perplexity_list[0], perplexity_list[1], perplexity_list[2]))


if __name__ == '__main__':
    # optimize the proper alpha
    LM_dic, best_alpha = train_LM()
    # define the train_file_set
    train_file_list = ['data/training.en', 'data/training.de', 'data/training.es']
    # implement the three model with best alpha
    for train_file in train_file_list:
        output_file = 'train_res/model.' + train_file.split('.')[1]
        model_implement(train_file, output_file, best_alpha)
    # generate sequence with 300 characters from our English model
    print('Generate sequence with 300 characters from our English model:')
    print(generate_from_LM('train_res/model.en') + '\n')
    # generate sequence with 300 characters from model-br.en
    print('Generate sequence with 300 characters from model-br.en:')
    print(generate_from_LM('data/model-br.en') + '\n')
    # define perplexity_list to restore the result of three models' perplexity to the test file
    perplexity_list = []
    # calculate three model's perplexity to the test file
    for train_file in train_file_list:
        model = 'train_res/model.' + train_file.split('.')[1]
        perplexity = compute_perplexity('data/test', model)
        perplexity_list.append(perplexity)
    # return the model with minimum perplexity
    res = train_file_list[perplexity_list.index(min(perplexity_list))]
    print('Perplexity:\nmodel.en: %s, model.de: %s, model.es: %s' % \
          (perplexity_list[0], perplexity_list[1], perplexity_list[2]))
    print('The language of test is model.%s' % (res.split('.')[1]))
