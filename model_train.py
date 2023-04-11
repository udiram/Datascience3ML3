from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build the convolutional network
model = Sequential()
model.add(Conv2D(1, kernel_size=(5, 5), activation='linear', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(50, 50)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print the number of parameters
print(model.summary())

# Train the model
history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    batch_size=64,
                    epochs=50,
                    verbose=1,
                    shuffle=True)
