import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [
    "avion", "automovil", "pajaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camion"
]

def main():
    (x_all, y_all), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_all = y_all.flatten()
    y_test = y_test.flatten()

    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.10, random_state=SEED, stratify=y_all
    )

    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10)
    ])

    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        augmentation,
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.20),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.30),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.40),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs("modelo", exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("modelo/cnn_cifar10.keras", monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=callbacks
    )

    x_test = x_test.astype("float32") / 255.0
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Loss de prueba: {loss:.4f}")
    print(f"Accuracy de prueba: {accuracy:.4f}")

    model.save("modelo/cnn_cifar10.keras")
    with open("modelo/class_names.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)
    print("Modelo y nombres de clases guardados en la carpeta modelo/")

if __name__ == "__main__":
    main()
