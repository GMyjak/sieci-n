import numpy
import random

class Perceptron:
    def __init__(self, size, f, theta):
        self.theta = theta
        self.f = f
        self.init_w(size)


    def init_w(self, size):
        self.w = []
        for i in range(size):
            value = 0
            value += (random.randint(0, 4) - 2) * 0.05
            self.w.append(value)


    def predict(self, vector):
        sum = numpy.dot(self.w, vector)
        return self.f(sum, self.theta)


    def learn(self, data, alpha):
        self.init_w(len(self.w))
        error_count = -1
        epoch_count = 0

        while error_count != 0:
            error_count = 0
            epoch_count += 1

            for idx, tuple in enumerate(data):
                xk = tuple[0]
                dk = tuple[1]
                zk = numpy.dot(self.w, xk)
                yk = self.f(zk, self.theta)
                err = dk - yk
                error_count += 1 if err != 0 else 0
                self.w = [self.w[idx] + alpha * err * xk[idx] for (idx, val) in enumerate(self.w)]

        print("Learning completed")
        print("Epochs: " + str(epoch_count))
        print("Weights: " + str(self.w))



def unipolar(value, theta):
    return 1 if value >= theta else 0


def bipolar(value, theta):
    return 1 if value >= theta else -1


if __name__ == "__main__":
    model = Perceptron(2, unipolar, 0.2)
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    model.learn(data, 0.1)
