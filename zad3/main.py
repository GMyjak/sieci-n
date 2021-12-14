import tensorflow as tf
from mnist import MNIST
from tensorflow import keras
import numpy as np

NUM_OF_EXPERIMENTS = 10


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


def test_conv(data = None, poolType='max', poolsize=(2,2), filters=12):
    if data == None:
        tr_data, tr_labels, vl_data, vl_labels = prepare_data('./zad3/data/ubyte')
    else:
        tr_data, tr_labels, vl_data, vl_labels = data

    tr_data = tr_data.reshape((tr_data.shape[0], 28, 28, 1))
    vl_data = vl_data.reshape((vl_data.shape[0], 28, 28, 1))

    if poolType == 'max':
        poolLayer = tf.keras.layers.MaxPooling2D(poolsize)
    elif poolType == 'avg':
        poolLayer = tf.keras.layers.AveragePooling2D(poolsize)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters, 3, input_shape=(28, 28, 1), activation='relu'),
        poolLayer,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy'])

    model.fit(tr_data, tr_labels, epochs=5, verbose=0)

    return model.evaluate(vl_data, vl_labels, verbose=0)


def test_mlp(data = None):
    if data == None:
        tr_data, tr_labels, vl_data, vl_labels = prepare_data('./zad3/data/ubyte')
    else:
        tr_data, tr_labels, vl_data, vl_labels = data

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784)),
        #tf.keras.layers.Dense(800, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy'])

    model.fit(tr_data, tr_labels, epochs=5, verbose=0)

    return model.evaluate(vl_data, vl_labels, verbose=0)


def test1():
    data = prepare_data('./zad3/data/ubyte')

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_mlp(data)
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("MLP:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data)
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("Conv:", acc, loss)


def test2():
    data = prepare_data('./zad3/data/ubyte')

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, 'max', (2,2))
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("MAX 2,2:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, 'max', (3,3))
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("MAX 3,3:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, 'max', (4,4))
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("MAX 4,4:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, 'avg', (2,2))
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("AVG 2,2:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, 'avg', (3,3))
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("AVG 3,3:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, 'avg', (4,4))
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("AVG 4,4:", acc, loss)


def test3():
    data = prepare_data('./zad3/data/ubyte')

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, filters=3)
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("CONV 3:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, filters=6)
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("CONV 6:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, filters=12)
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("CONV 12:", acc, loss)

    acc = 0
    loss = 0
    for i in range(NUM_OF_EXPERIMENTS):
        score = test_conv(data, filters=20)
        acc += score[0]
        loss += score[1]

    acc /= NUM_OF_EXPERIMENTS
    loss /= NUM_OF_EXPERIMENTS

    print("CONV 20:", acc, loss)

def main():
    test3()


if __name__ == '__main__':
    main()
