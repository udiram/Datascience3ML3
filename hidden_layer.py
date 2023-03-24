from keras.utils import to_categorical

# Convert labels to one-hot encoding
Y_c = to_categorical(Y, 2)

# Define model architecture
model = Sequential()
model.add(Dense(8, input_shape=(2,), activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=SGD(lr=0.15), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X, Y_c, epochs=400, batch_size=128)
