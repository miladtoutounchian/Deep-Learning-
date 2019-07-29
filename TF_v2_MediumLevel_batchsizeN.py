import tensorflow as tf
import numpy as np

# This code needs TF 2.0

inputs = tf.keras.Input(shape=(4,))
hidden = tf.keras.layers.Dense(3, activation=tf.nn.sigmoid)(inputs)
outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Instantiate a logistic loss function that expects integer targets.
mse = tf.keras.losses.MeanSquaredError()

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)

# Input array
X_data = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])

# Output
y_data = np.array([[1.0], [1.0], [0.0]])

x = tf.convert_to_tensor(X_data, dtype=tf.float64)
y = tf.convert_to_tensor(y_data, dtype=tf.float64)

for _ in range(2000):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = mse(y, pred)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))


x = tf.convert_to_tensor(np.array([X_data[2]]), dtype=tf.float64)
print(model(x).numpy())