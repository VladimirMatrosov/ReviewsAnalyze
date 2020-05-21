from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd


def read_csv(path):
    return pd.read_csv(path, header=None, names=['Class', 'Review'])


class DatasetHelper:

    def __init__(self, files_path, count_words=10000, max_len=100, name_train='train.csv', name_test='test.csv'):
        train = read_csv(files_path + name_train)
        test = read_csv(files_path + name_test)

        reviews_train = train['Review']
        self.y_train = train['Class'] - 1
        reviews_test = test['Review']
        self.y_test = test['Class'] - 1

        self.tokenizer = Tokenizer(num_words=count_words)
        self.tokenizer.fit_on_texts(reviews_train)
        x_train = self.tokenizer.texts_to_sequences(reviews_train)
        x_test = self.tokenizer.texts_to_sequences(reviews_test)

        self.x_train = pad_sequences(x_train, maxlen=max_len)
        self.x_test = pad_sequences(x_test, maxlen=max_len)
        self.max_len = max_len
        self.count_words = count_words

    def texts_to_sequences(self, texts):
        return pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.max_len)
