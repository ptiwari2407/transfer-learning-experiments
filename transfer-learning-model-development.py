# This is an example of transfer learning

from sklearn.datasets import make_blobs
from numpy import where
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD



def samples_for_seed(seed):
    X, y = make_blobs(n_samples=1000, centers = 3, n_features =2, cluster_std=2, random_state=seed)
    y = to_categorical(y)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, : ]
    trainY, testY = y[:n_train], y[n_train:]
    return trainX, trainY, testX, testY


# def plot_samples(X, y, classes = 3):
#     for i in range(classes):
#         samples_ix = where(y==i)
#         plt.scatter(X[samples_ix, 0], X[samples_ix, 1])
#
# n_problems = 2
# for i in range(1, n_problems+1):
#     plt.subplot(210 + i)
#     X, y = samples_for_seed(i)
#     plot_samples(X, y)
#
# plt.show()

def fit_model(trainX, trainY, testX, testY):
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs = 100, verbose=0 )
    return model, history

def summarize_model(model, history, trainX, trainY, testX, testY):
    _, train_acc = model.evaluate(trainX, trainY, verbose=0)
    _, test_acc = model.evaluate(testX, testY, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()

    plt.show()

# For model 1
trainX, trainY, testX, testY = samples_for_seed(1)
model, history = fit_model(trainX, trainY, testX, testY)
summarize_model(model, history, trainX, trainY, testX, testY)

model.save('model.h5')

# Now training the same model for problem 2 and checking the relevant performances.
trainX, trainY, testX, testY = samples_for_seed(2)
model, history = fit_model(trainX, trainY, testX, testY)
summarize_model(model, history, trainX, trainY, testX, testY)