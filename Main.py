from DatasetHelper import DatasetHelper
from ReviewsNeuronet import show_train_process, Neuronet
import numpy as np

dataset = DatasetHelper('D:/yelp_review_polarity_csv/')
x_train = dataset.x_train
y_train = dataset.y_train
x_test = dataset.x_test
y_test = dataset.y_test

neuronet = Neuronet(dataset.count_words, dataset.max_len)
history = neuronet.train(x_train, y_train, 'reviews_analyzer.h5')
show_train_process(history)
neuronet.load('reviews_analyzer.h5')
scores = neuronet.evaluate(x_test, y_test)
print("Доля верных ответов на тестовой выборке: ", round(scores[1] * 100, 4))

positive_review = "Really cool app for spending free time with friends"
negative_review = "It's bad for your password, you forget it and you can't use a normal one you use"

reviews = [positive_review, negative_review]
review_sequences = dataset.texts_to_sequences(reviews)
for i, review in enumerate(reviews):
    print(review)
    score = neuronet.predict(np.array([review_sequences[i]]))
    print(score)