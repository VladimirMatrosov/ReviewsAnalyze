from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Dropout, LSTM, Flatten, MaxPooling1D
from kerastuner.tuners import Hyperband
from src.ImdbHelper import ImdbHelper

imdb = ImdbHelper()
x_train = imdb.x_train
y_train = imdb.y_train
x_test = imdb.x_test
y_test = imdb.y_test


def build_model_conv(hp):
    model = Sequential()
    model.add(Embedding(
        imdb.count_words,
        hp.Int('vect_size', min_value=2, max_value=64, step=4),
        input_length=imdb.max_len
    ))
    for i in range(hp.Int('num_layers', min_value=1, max_value=5, step=1)):
        model.add(Conv1D(
            hp.Int('conv_units_' + str(i), min_value=128, max_value=256, step=16),
            5,
            activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(
        hp.Int('dense_units', min_value=64, max_value=256, step=32),
        activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_model_lstm(hp):
    model = Sequential()
    model.add(Embedding(
        imdb.count_words,
        hp.Int('vect_size', min_value=2, max_value=64, step=4),
        input_length=imdb.max_len
    ))
    count_layers = hp.Int('num_layers', min_value=1, max_value=5, step=1)
    for i in range(count_layers):
        model.add(LSTM(
            hp.Int('lstm_units' + str(i), min_value=64, max_value=256, step=16),
            return_sequences=i != count_layers - 1))
        model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_tuner(is_lstm):
    if is_lstm:
        return Hyperband(
            build_model_lstm,
            objective='val_accuracy',
            factor=2,
            max_epochs=5,
            directory='lstm'
        )
    else:
        return Hyperband(
            build_model_conv,
            objective='val_accuracy',
            factor=2,
            max_epochs=5,
            directory='conv'
        )


tuner = build_tuner(True)

tuner.search(
    x_train,
    y_train,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

tuner.results_summary()

results = tuner.get_best_models(3)
for result in results:
    result.summary()
    scores = result.evaluate(x_test, y_test, verbose=1)
    print("Доля верных ответов на тестовом наборе данных: ", round(scores[1] * 100, 4))
