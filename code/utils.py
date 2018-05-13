import threading
import pandas as pd
import os
import numpy as np

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

def create_data_csv(path):
    path_to_features = path
    classes = ['get', 'mov', 'ono', 'palm', 'put', 'pao']

    features_dir = os.walk(path_to_features)
    data_to_write = []

    for dir in features_dir:

        for file in dir[2]:

            path_to_file = path_to_features + file
            target = None

            for i in range(len(classes)):
                if file.startswith(classes[i]):
                    target = i
                    break

            if target is None:
                continue

            data_to_write.append([path_to_file, target])

    dataframe = pd.DataFrame(data_to_write)
    dataframe.to_csv(path_to_features + 'data.csv', sep=',')

def hot_one(cls):
    classes = ['get', 'mov', 'ono', 'palm', 'put', 'pao']
    encode_template = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if isinstance(cls, str):
        if cls in classes:
            idx = classes.index(cls)
            encode_template[idx] = 1.0
            return encode_template

        print("Bad string class indetifier!")

    elif isinstance(cls, list):
        if len(cls) == len(classes) and 1.0 in cls:
            return classes[cls.index(1.0)]

        print("Bad input hot-one list!")

    elif isinstance(cls, int):
        if cls > -1 and cls < len(classes):
            encode_template[cls] = 1.0
            return encode_template

        print("Bad int class indetifier")

    return None

def get_sequence(path_to_file):
    if os.path.isfile(path_to_file):
        return np.load(path_to_file)
    return None


def get_all_svm(path_to_csv, data_type):
    dataframe = pd.read_csv(path_to_csv)
    dataframe = dataframe[dataframe['2']==data_type]
    X, y = [], []

    for idx, row in dataframe.iterrows():
        sequence = get_sequence(row['0'])

        if sequence is None:
            raise ValueError("Can't find sequence. Did you extracted them?")

        cls = int(row['1'])

        if cls is None:
            raise ValueError("Can't encode class!")

        X.append(sequence)
        y.append(cls)

    return np.array(X), np.array(y)


def get_all_sequences_in_memory(type_data, path_to_csv='../data/features/data.csv'):

    dataframe = pd.read_csv(path_to_csv)
    items = dataframe[dataframe['2'] == type_data]
    X, y = [], []

    for idx, row in items.iterrows():
        sequence = get_sequence(row['0'])

        if sequence is None:
            raise ValueError("Can't find sequence. Did you extracted them?")

        cls = hot_one(row['1'])

        if cls is None:
            raise ValueError("Can't encode class!")

        X.append(sequence)
        y.append(cls)

    return np.array(X), np.array(y)

def split_data(path_to_csv, ratio=0.8):

    dataframe = pd.read_csv(path_to_csv)
    cnt_data = dataframe.shape[0]
    target_vector = [ None for i in range(cnt_data)]
    cnt_train = int(cnt_data* ratio)
    j = 1

    for i in np.random.permutation(cnt_data):
        target_vector[i] = 'train' if j < cnt_train else 'test'
        j += 1

    dataframe['2'] = pd.Series(target_vector, index=dataframe.index)
    dataframe.to_csv(path_to_csv)

def max_sequence_len(data, axis=0):

    max_sq_len = 0

    for darr in data:
        if max_sq_len < darr.shape[axis]:
            max_sq_len = darr.shape[axis]

    return max_sq_len