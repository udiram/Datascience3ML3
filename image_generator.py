import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

def generate_stripe_image(size, stripe_nr, vertical=True):
    img = np.zeros((size, size, 1), dtype="uint8")
    for i in range(stripe_nr):
        x, y = np.random.randint(0, size, 2)
        l = np.int(np.random.randint(y, size, 1))
        if vertical:
            img[y:l, x, 0] = 255
        else:
            img[x, y:l, 0] = 255
    return img

def generate_data(size, stripe_nr, num_samples):
    X = np.zeros((num_samples, size, size, 1))
    Y = np.zeros((num_samples, 2))
    for i in range(num_samples):
        if i % 2 == 0:
            img = generate_stripe_image(size, stripe_nr, vertical=True)
            Y[i, 0] = 1
        else:
            img = generate_stripe_image(size, stripe_nr, vertical=False)
            Y[i, 1] = 1
        X[i] = img / 255.0
    return X, Y

# Generate the training and validation data
X_train, Y_train = generate_data(50, 10, 1000)
X_val, Y_val = generate_data(50, 10, 1000)

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
