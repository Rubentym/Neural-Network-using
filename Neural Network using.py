import tensorflow as tf

# Sample data
X = tf.constant([[1.0, 2.0]])
y_true = tf.constant([[0.0, 1.0]])

# Define a neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='sgd', loss='mse')

# Train the model
model.fit(X, y_true, epochs=1000, verbose=0)

# Predictions
y_pred = model.predict(X)
print("Predictions:", y_pred)
