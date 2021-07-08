import pickle
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoad(object):
    def __init__(self, vocab_path=None, data_path=None):
        with open(vocab_path, 'rb') as vocab_inp:
            self.word2id = pickle.load(vocab_inp)
            self.id2word = pickle.load(vocab_inp)
            self.tag2id = pickle.load(vocab_inp)
            self.id2tag = pickle.load(vocab_inp)

        with open(data_path, 'rb') as inp:
            self.X = pickle.load(inp)
            self.Y = pickle.load(inp)

    def buildTraindata(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        print("train size: %d valid size: %d    test size: %d" % (x_train.shape[0], x_valid.shape[0], x_test.shape[0]))

        train_batcher = BatchGenerator(x_train, y_train, shuffle=True)
        valid_batcher = BatchGenerator(x_valid, y_valid, shuffle=False)
        test_batcher = BatchGenerator(x_test, y_test, shuffle=False)
        print("finish create dataset!")
        return train_batcher, valid_batcher, test_batcher

class BatchGenerator(object):
    def __init__(self, x_inputs, y_inputs, shuffle=False):
        if type(x_inputs) != np.ndarray:
            x_inputs = np.asarray(x_inputs)
        if type(y_inputs) != np.ndarray:
            y_inputs = np.asarray(y_inputs)
        self._x_inputs = x_inputs
        self._y_inputs = y_inputs

        self.count_for_epoch = 0
        self.shuffle = shuffle
        self.data_size = x_inputs.shape[0]
        if self.shuffle:
            new_index = np.random.permutation(self.data_size)
            self._x_inputs = self._x_inputs[new_index]
            self._y_inputs = self._y_inputs[new_index]

    @property
    def x_inputs(self):
        return self._x_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    def next_batch(self, batch_size):
        start_ind = self.count_for_epoch
        self.count_for_epoch += batch_size
        if self.count_for_epoch > self.data_size:
            start_ind = 0
            self.count_for_epoch = batch_size
            if self.shuffle:
                new_index = np.random.permutation(self.data_size)
                self._x_inputs = self._x_inputs[new_index]
                self._y_inputs = self._y_inputs[new_index]
        end_ind = self.count_for_epoch
        return self._x_inputs[start_ind: end_ind], self._y_inputs[start_ind, end_ind]