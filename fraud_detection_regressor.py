import numpy as np
from urllib.request import urlopen
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
raw_data = urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)

X = dataset[:, [1, 3]]
Y = dataset[:, 4]

# plot the data points
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Skewness of wavelet transformed image')
plt.ylabel('Entropy of wavelet transformed image')
plt.show()

# define the model
model = Sequential()
model.add(Dense(1, batch_input_shape=(None, 2), activation='sigmoid'))

# compile the model
sgd = SGD(lr=0.15)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train the model
history = model.fit(X, Y, epochs=400, batch_size=128, verbose=1)

# plot the accuracy and loss history
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
