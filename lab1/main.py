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


class Adaline:
    def __init__(self, size, f, theta):
        self.theta = theta
        self.f = f
        self.init_w(size)


    def init_w(self, size):
        self.w = []
        for i in range(size):
            value = 0
            value += (random.randint(0, 4) - 2) * 0.2
            self.w.append(value)


    def predict(self, vector):
        sum = numpy.dot(self.w, vector)
        return sum


    def predict_with_function(self, vector):
        sum = self.predict(vector)
        return self.f(sum, self.theta)


    def ms_error(self, data):
        sum = 0
        
        for tuple in data:
            err = tuple[1] - self.predict(tuple[0])
            err = err * err
            sum += err

        sum = sum / len(data)
        return sum


    def learn(self, data, alpha, threshold):
        self.init_w(len(self.w))
        error = self.ms_error(data)

        while error > threshold:
            # epoch
            for data_idx, tuple in enumerate(data):
                err = tuple[1] - self.predict(tuple[0])
                self.w = [self.w[w_idx] + 2 * alpha * err * data[data_idx][0][w_idx] for (w_idx, val) in enumerate(self.w)]
            error = self.ms_error(data)
            print("Error: " + str(error))


class AdalineBias:
    def __init__(self, size, f):
        self.bias = 0
        self.f = f
        self.init_w(size)


    def init_w(self, size):
        self.w = []
        for i in range(size):
            value = 0
            value += (random.randint(0, 4) - 2) * 0.1
            self.w.append(value)


    def predict(self, vector):
        sum = numpy.dot(self.w, vector)
        sum += self.bias
        return sum


    def predict_with_function(self, vector):
        sum = self.predict(vector)
        return self.f(sum)


    def ms_error(self, data):
        sum = 0
        
        for tuple in data:
            err = tuple[1] - self.predict(tuple[0])
            err = err * err
            sum += err

        sum = sum / len(data)
        return sum


    def learn(self, data, alpha, threshold):
        self.init_w(len(self.w))
        error = self.ms_error(data)
        epochs = 0

        while error > threshold:
            # epoch
            epochs += 1
            for data_idx, tuple in enumerate(data):
                err = tuple[1] - self.predict(tuple[0])
                self.w = [self.w[w_idx] + 2 * alpha * err * data[data_idx][0][w_idx] for (w_idx, val) in enumerate(self.w)]
                self.bias = self.bias + 2 * alpha * err
            error = self.ms_error(data)
            
        print("Learning completed")
        print("Epochs: " + str(epochs))
        print("Error: " + str(error))


def unipolar(value, theta = 0):
    return 1 if value >= theta else 0


def bipolar(value, theta = 0):
    return 1 if value >= theta else -1


if __name__ == "__main__":
    # model = Perceptron(2, unipolar, 0.2)
    # data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    # model.learn(data, 0.1)

    model = AdalineBias(2, bipolar)
    data = [([-1, -1], -1), ([-1, 1], -1), ([1, -1], -1), ([1, 1], 1)]
    model.learn(data, 0.05, 0.3)
    # print("AND(0,1): " + str(model.predict([-1,1])))
    # print("AND(1,1): " + str(model.predict([1,1])))
    # print("AND(0,0): " + str(model.predict([-1,-1])))
    # print("AND(1,0): " + str(model.predict([1,-1])))

    print("AND(1,1): " + str(model.predict_with_function([1,1])))
    print("AND(0,1): " + str(model.predict_with_function([-1,1])))
    print("AND(0,0): " + str(model.predict_with_function([-1,-1])))
    print("AND(1,0): " + str(model.predict_with_function([1,-1])))
