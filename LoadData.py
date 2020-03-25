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
