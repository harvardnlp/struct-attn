#!/usr/bin/env python

""" Create data for QA bAbI tasks """

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import copy
import operator
import json
import glob, os
import csv
import re

START_TOKEN = "<s>"
END_TOKEN = "</s>"
PAD_TOKEN = "PADDING"
RARE_TOKEN = "RARE"

MAX_QUESTION = 0
MAX_FACT = 0

NUM_TASKS = 20

args = {}

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def init_vocab():
    return { RARE_TOKEN : 1, PAD_TOKEN : 2, START_TOKEN: 3, END_TOKEN: 4 }


def write_dict(file, dict):
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1))
    writer = csv.writer(open(file, 'wb'))
    for key, value in sorted_dict:
       writer.writerow([key, value])


def get_vocab(folder, word_to_idx = {}):
    if len(word_to_idx) == 0:
        word_to_idx = init_vocab()

    idx = len(word_to_idx) + 1
    owd = os.getcwd()
    os.chdir(folder)
    for infile in glob.glob("*.txt"):
        with open(infile) as inf:
            for line in inf:
                parts = line.lower().replace('.','').strip().split('?')
                tokens = parts[0].split(' ')
                for i in np.arange(1, len(tokens)):
                    if tokens[i] not in word_to_idx and not is_number(tokens[i]):
                        word_to_idx[tokens[i]] = idx
                        idx += 1
                if len(parts) > 1:
                    answer = parts[1].strip().split('\t')[0]
                    if answer not in word_to_idx:
                        word_to_idx[answer.lower()] = idx
                        idx += 1
    os.chdir(owd)
    return word_to_idx


def normalize(data, word_to_idx):
    for t in np.arange(NUM_TASKS) + 1:
        questions = data[t]['questions']
        answers = data[t]['answers']
        facts = data[t]['facts']

        pad_idx = word_to_idx[PAD_TOKEN]
        norm_questions = np.ones((len(questions), MAX_QUESTION)) * pad_idx
        norm_answers = np.array(answers)
        norm_facts = np.zeros((len(facts), MAX_FACT))

        for i in range(len(questions)):
            norm_questions[i, : len(questions[i])] = questions[i]
        for i in range(len(facts)):
            norm_facts[i, : len(facts[i])] = facts[i]

        data[t]['questions'] = norm_questions
        data[t]['answers'] = norm_answers
        data[t]['facts'] = norm_facts


def process_answer(label, task_no, word_to_idx):
    '''
    Process output label for each sentence depending on the task number.
    '''
    parts = label.strip().split('\t')
    # TODO: ignore caps in answer?
    return word_to_idx[parts[0].lower()], parts[1].split(' ')


def process_task_max(file, task_no, task_to_max, word_to_idx):
    pad_idx = word_to_idx[PAD_TOKEN]

    # first loop to get max number of sentences for each story
    if task_no not in task_to_max:
        task_to_max[task_no] = {}
        task_to_max[task_no]['max_line_num'] = 0
        task_to_max[task_no]['max_line_len'] = 0

    with open(file) as inf:
        num_story_line = 0
        for line in inf:
            line_info = re.match('([0-9].*?)\ (.*)', line)
            line_no = int(line_info.group(1)) # line number
            if line_no == 1:
                task_to_max[task_no]['max_line_num'] = max(task_to_max[task_no]['max_line_num'], num_story_line)
                num_story_line = 0
            line_length = len(line_info.group(2).strip().split(' ')) # rest of line
            if line_length > task_to_max[task_no]['max_line_len']:
                task_to_max[task_no]['max_line_len'] = line_length
            if '?' not in line:
                num_story_line += 1
        task_to_max[task_no]['max_line_num'] = max(task_to_max[task_no]['max_line_num'], num_story_line)


def process(file, task_no, task_to_max, word_to_idx):
    '''
    Process one single QA file (either train or test).
    '''
    global MAX_QUESTION
    global MAX_FACT

    pad_idx = word_to_idx[PAD_TOKEN]

    # first loop to get max number of sentences for each story
    MAX_LINE_NUM = task_to_max[task_no]['max_line_num']
    MAX_LINE_LEN = task_to_max[task_no]['max_line_len']
    NUM_QUESTION = 0
    with open(file) as inf:
        for line in inf:
            if '?' in line:
                NUM_QUESTION += 1
    
    all_stories = np.ones((NUM_QUESTION, MAX_LINE_NUM, MAX_LINE_LEN + 2)) * pad_idx
    all_line_nos = np.zeros((NUM_QUESTION, MAX_LINE_NUM))
    all_questions = []
    all_answers = []
    all_facts = []
    current_story = []
    current_line_nos = []
    with open(file) as inf:
        for line in inf:
            line_info = re.match('([0-9].*?)\ (.*)', line)
            line_no = int(line_info.group(1)) # line number
            line_data = line_info.group(2) # rest of line
            question_no = len(all_questions)

            parts = line_data.split('?')

            # parse the first part, either statement or question
            statement = parts[0].strip().replace('.', '').split(' ')
            words = [START_TOKEN] + [w.lower() for w in statement] + [END_TOKEN]

            if line_no == 1:
                current_story = []
                current_line_nos = []

            if len(parts) > 1: # is a question
                all_stories[question_no][
                    MAX_LINE_NUM - len(current_story) : MAX_LINE_NUM] = current_story
                all_line_nos[question_no][
                    MAX_LINE_NUM - len(current_line_nos) : MAX_LINE_NUM] = current_line_nos
                    
                MAX_QUESTION = max(MAX_QUESTION, len(words))
                all_questions.append([word_to_idx[w] for w in words]) # append to question list

                answer, facts = process_answer(parts[1], task_no, word_to_idx)
                all_answers.append(answer)

                MAX_FACT = max(MAX_FACT, len(facts))
                all_facts.append(facts)
            else: # is not a question
                current_story.append([word_to_idx[words[i]] if i < len(words) else pad_idx 
                    for i in range(MAX_LINE_LEN + 2)])
                current_line_nos.append(line_no)
                

    return {
    'stories': all_stories, 'questions': all_questions,
    'answers': all_answers, 'facts': all_facts,
    'linenos': all_line_nos }


def process_files(folder, word_to_idx):
    '''
    Process all QA files in specified folder.
    '''
    trains = {}
    tests = {}
    tasks = ['' for t in range(NUM_TASKS)] # task names

    owd = os.getcwd()
    os.chdir(folder)

    task_to_max = {} # keep track of max sent length, sentence count etc... for each task

    for infile in glob.glob("*.txt"):
        file_info = re.match('qa(.*)_(.*)_(.*).txt', infile)
        task_no = int(file_info.group(1)) # from 1 to 20
        process_task_max(infile, task_no, task_to_max, word_to_idx)

    for infile in glob.glob("*.txt"):
        file_info = re.match('qa(.*)_(.*)_(.*).txt', infile)
        task_no = int(file_info.group(1)) # from 1 to 20
        task_name = file_info.group(2) # e.g. yes-no-questions, basic-induction
        task_data_type = file_info.group(3) # train or test

        # process data
        processed_data = process(infile, task_no, task_to_max, word_to_idx)
        if task_data_type == 'train':
            trains[task_no] = processed_data
        else:
            tests[task_no] = processed_data

        # add the task name
        tasks[task_no - 1] = task_name

    os.chdir(owd)

    return trains, tests, tasks


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-vocabsize', help="vocabsize",
                        type=long,default=500000,required=False)
    parser.add_argument('-dir', help="data directory",
                        type=str,default='babi_data/en/',required=False)
    args = parser.parse_args(arguments)

    word_to_idx = get_vocab(args.dir)
    write_dict('word_to_idx.csv', word_to_idx) # for debugging purposes

    trains, tests, tasks = process_files(args.dir, word_to_idx)
    normalize(trains, word_to_idx)
    normalize(tests, word_to_idx)

    for t in np.arange(NUM_TASKS) + 1:
        filename = 'qa{0:02d}.hdf5'.format(t)
        with h5py.File(filename, "w") as f:
            f['train_stories'] = trains[t]['stories']
            f['train_linenos'] = trains[t]['linenos']
            f['train_questions'] = trains[t]['questions']
            f['train_answers'] = trains[t]['answers']
            f['train_facts'] = trains[t]['facts']

            f['test_stories'] = tests[t]['stories']
            f['test_linenos'] = tests[t]['linenos']
            f['test_questions'] = tests[t]['questions']
            f['test_answers'] = tests[t]['answers']
            f['test_facts'] = tests[t]['facts']

            f['nwords'] = np.array([len(word_to_idx)], dtype=np.int32)

            f['idx_pad'] = np.array([word_to_idx[PAD_TOKEN]], dtype=np.int32)
            f['idx_rare'] = np.array([word_to_idx[RARE_TOKEN]], dtype=np.int32)
            f['idx_start'] = np.array([word_to_idx[START_TOKEN]], dtype=np.int32)
            f['idx_end'] = np.array([word_to_idx[END_TOKEN]], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
