print("importing libraries...")
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pickle
from sys import argv
import gzip
from matplotlib import pyplot as plt

np.random.seed(5)
print_length = 10000

def get_data():
    x = 0
    print("creating featuresets...")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load()
        training_inputs = [np.reshape(x, (1, 784)).astype('float64') for x in training_data[0]]
        training_results = [vectorized_result(y).astype('float64') for y in training_data[1]]

        validation_inputs = [np.reshape(x, (1, 784)) for x in validation_data[0]]
        validation_results = [vectorized_result(y) for y in validation_data[1]]

        test_inputs = [np.reshape(x, (1, 784)) for x in test_data[0]]
        test_results = [vectorized_result(y) for y in test_data[1]]
    return training_inputs, training_results, validation_inputs, validation_results, test_inputs, test_results

def vectorized_result(j):
    e = np.zeros((1, 10))
    e[0][j] = 1.0
    return e

def interpret_output(y):
    return np.argmax(y)

train_x, train_y, val_x, val_y, test_x, test_y = get_data()

model = Sequential()
model.add(Dense(50, input_dim=784, activation='relu', use_bias=True, bias_initializer='zeros'))
model.add(Dense(40, activation='relu', use_bias=True, bias_initializer='zeros'))
model.add(Dense(10, activation='sigmoid', use_bias=True, bias_initializer='zeros'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.vstack(train_x), np.vstack(train_y), epochs=10, batch_size=10)
prediction = model.predict(np.vstack(test_x), batch_size=100)
accuracy = model.evaluate(np.vstack(test_x), np.vstack(test_y))[1] * 100
print(f"{str(accuracy)[:4]}%")

if "print" in argv:
    for image in range(print_length):
        guess = interpret_output(prediction[image])
        real = interpret_output(test_y[image])
        plt.imshow(test_x[image].reshape(28, 28), cmap="gray")
        plt.title(f"Guess: {guess}, Real Result: {real}")
        plt.savefig(f"tests/test{image}.png")
        plt.close('all')
        print(f"image {image + 1} out of {print_length}", end="\r")