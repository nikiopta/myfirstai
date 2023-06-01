import tensorflow as tf
from tensorflow import keras
from keras import layers
tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
model = keras.Sequential([
    keras.Input((28, 28, 1)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
)
model = keras.models.load_model("saved_model")
print(model.summary())
model.fit(x_train, y_train, batch_size=64, epochs=1, callbacks=[keras.callbacks.ModelCheckpoint("saved_model")],)
model.evaluate(x_test, y_test, callbacks=[keras.callbacks.ModelCheckpoint("saved_model")])