from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def show_train_process(history):
    plt.plot(history.history["accuracy"], label="Доля верных ответов на обучающем наборе")
    plt.plot(history.history["val_accuracy"], label="Доля верных ответов на валидационной выборке")
    plt.xlabel("Эпоха обучения")
    plt.ylabel("Доля верных ответов")
    plt.legend()
    plt.show()


class Neuronet:

    def __init__(self, count_words, max_len):
        self.model = Sequential()
        self.model.add(Embedding(count_words, 34, input_length=max_len))
        self.model.add(Conv1D(160, 5, activation='relu'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Dropout(0.25))
        self.model.add(Conv1D(160, 5, activation='relu'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(192, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, name_model):
        checkpoint = ModelCheckpoint(name_model, monitor='val_accuracy', save_best_only=True, verbose=1)
        return self.model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.2,
            callbacks=[checkpoint],
            verbose=1
        )

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=1)

    def load(self, name_model):
        self.model = load_model(name_model)
        self.model.summary()

    def predict(self, sequence_text):
        return self.model.predict(sequence_text, verbose=1)
