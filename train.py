import pandas as pd
import numpy as np
import pickle
import csv
import pymorphy2
import re
import keras.utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

ma = pymorphy2.MorphAnalyzer()

def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols
    text = " ".join(ma.parse(str(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)
    # text = text.encode("utf-8")

    return str(text)

def load_data_from_arrays(strings, labels, train_test_split=0.9):
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))

    x_train = strings[test_size:]
    y_train = labels[test_size:]

    x_test = strings[:test_size]
    y_test = labels[:test_size]

    return x_train, y_train, x_test, y_test


data = pd.read_csv('train_data.csv', sep=';', header=None)
descriptions = data.apply(lambda x: clean_text(x[1]), axis=1)
categories = data[2]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions.tolist())

textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, categories, train_test_split=0.9)

maxSequenceLength = 0
for desc in descriptions.tolist():
    words = len(desc.split())
    if words > maxSequenceLength:
        maxSequenceLength = words
# Максимальное количество слов в самом длинном описании аккаунта

total_unique_words = len(tokenizer.word_counts)
vocab_size = total_unique_words + 1
# Всего уникальных слов во всех описаниях аккаунтов

tokenizer = Tokenizer(num_words=vocab_size, lower = False)
tokenizer.fit_on_texts(descriptions)

# X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)


# y_train = keras.utils.to_categorical(y_train, num_categories)
# y_test = keras.utils.to_categorical(y_test, num_categories)
# Если категорий больше двух, преобразуем столбец категорий в бинарную матрицу


# model = Sequential()
# model.add(Dense(512, input_shape=(num_words,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print(model.summary())
#
# history = model.fit(X_train, y_train,
#                     batch_size=32,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)
# MLP - модель

batch_size = 32
epochs = 3

model = Sequential()
model.add(Embedding(vocab_size, maxSequenceLength))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))

# LSTM - модель (результаты лучше)

score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))

for i in range(len(X_test)):
    prediction = model.predict(np.array([X_test[i]]))
    print(data[1][i], y_test[i], prediction, '\n\n\n\n\n\n\n\n\n')
# Выводим описание аккаунта, нашу оценку и оценку модели
