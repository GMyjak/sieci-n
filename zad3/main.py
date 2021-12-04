import tensorflow as tf
from mnist import MNIST
from tensorflow import keras
import numpy as np


label_matrix = [[1,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]


def prepare_data(path):
    mndata = MNIST(path)
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    tr_images = np.array(tr_images) / 255
    vl_images = np.array(vl_images) / 255

    tr_labels = np.array(tr_labels)
    vl_labels = np.array(vl_labels)

    return tr_images, tr_labels, vl_images, vl_labels


def test_conv():
    tr_data, tr_labels, vl_data, vl_labels = prepare_data('./zad3/data/ubyte')

    tr_data = tr_data.reshape((tr_data.shape[0], 28, 28, 1))
    vl_data = vl_data.reshape((vl_data.shape[0], 28, 28, 1))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(12, 3, input_shape=(28, 28, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy'])

    model.fit(tr_data, tr_labels, epochs=5)

    model.evaluate(vl_data, vl_labels, verbose=1)


def test_mlp():
    tr_data, tr_labels, vl_data, vl_labels = prepare_data('./zad3/data/ubyte')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(800, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

    model.compile(optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy'])

    model.fit(tr_data, tr_labels, epochs=5)

    model.evaluate(vl_data, vl_labels, verbose=1)


def main():
    test_conv()


if __name__ == '__main__':
    main()
