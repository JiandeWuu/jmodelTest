import os
import csv
import pickle

from jmodel.np import *

def read_csv(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as csvfile:
        return np.array([row for row in csv.DictReader(csvfile)])

# 製作運算矩陣
def parse_data(data, x_targets, y_target = 'CITY'):
    x = np.array([[float(row[_element]) for _element in x_targets ] for row in data ])
    y = np.array([row[y_target] for row in data])
    unique = np.unique(y)
    y = np.array([list(unique).index(row) for row in y])

    return x, y, unique

def load(file_path, features = ['LON', 'LAT'], y_target = 'CITY'):
    vocab_path = os.path.splitext(file_path)[0] + "_" + str(features) + '.pkl'
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            x, y, unique = pickle.load(f)
        return x, y, unique

    rows = read_csv(file_path)
    data = np.array(rows)
    x, y, unique = parse_data(data, features, y_target = y_target)
    
    with open(vocab_path, 'wb') as f:
        pickle.dump((x, y, unique), f)
    return x, y, unique



dataset_dir = os.path.dirname(os.path.abspath(__file__))

def load_vocab(file_path):
    vocab_path = os.path.splitext(file_path)[0] + ".pkl"

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}

    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(file_name):
    
    file_path = dataset_dir + '/' + file_name
    word_to_id, id_to_word = load_vocab(file_path)

    save_path = os.path.splitext(file_path)[0] + ".npy"
    if os.path.exists(save_path):
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    corpus = np.array([word_to_id[w] for w in words])
    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word
