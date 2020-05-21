from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class ImdbHelper:

    def __init__(self, count_words=10000, max_len=200):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=count_words)
        data = np.concatenate((x_train, x_test))
        targets = np.concatenate((y_train, y_test))
        self.x_train = pad_sequences(data[:40000], maxlen=max_len)
        self.y_train = targets[:40000]
        self.x_test = pad_sequences(data[40000:], maxlen=max_len)
        self.y_test = targets[40000:]
        self.count_words = count_words
        self.max_len = max_len
