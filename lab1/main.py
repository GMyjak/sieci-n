import numpy
import random

class Perceptron:
    def __init__(self, size, f, theta, max_init_weight = 0.2):
        self.max_init_weight = max_init_weight
        self.theta = theta
        self.f = f
        self.init_w(size)


    def init_w(self, size):
        self.w = []
        for i in range(size):
            value = 0
            value += (random.random() * 2 - 1) * self.max_init_weight
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

        # print("Learning completed")
        # print("Epochs: " + str(epoch_count))
        # print("Weights: " + str(self.w))

        return epoch_count


class PerceptronBias:
    def __init__(self, size, f, theta, max_init_weight = 0.2):
        self.max_init_weight = max_init_weight
        self.bias = 0
        self.theta = theta
        self.f = f
        self.init_w(size)


    def init_w(self, size):
        self.w = []
        for i in range(size):
            value = 0
            value += (random.random() * 2 - 1) * self.max_init_weight
            self.w.append(value)


    def predict(self, vector):
        sum = numpy.dot(self.w, vector)
        sum += self.bias
        return sum

    def predict_with_function(self, vector):
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
                zk = self.predict(xk)
                yk = self.f(zk, self.theta)
                err = dk - yk
                error_count += 1 if err != 0 else 0
                self.w = [self.w[idx] + alpha * err * xk[idx] for (idx, val) in enumerate(self.w)]
                self.bias = self.bias + alpha * err

        # print("Learning completed")
        # print("Epochs: " + str(epoch_count))
        # print("Weights: " + str(self.w))

        return epoch_count


class Adaline:
    def __init__(self, size, f, theta, max_init_weight = 0.2):
        self.max_init_weight = max_init_weight
        self.theta = theta
        self.f = f
        self.init_w(size)


    def init_w(self, size):
        self.w = []
        for i in range(size):
            value = 0
            value += (random.random() * 2 - 1) * self.max_init_weight
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
        epochs = 0

        while error > threshold:
            # epoch
            epochs += 1
            for data_idx, tuple in enumerate(data):
                err = tuple[1] - self.predict(tuple[0])
                self.w = [self.w[w_idx] + 2 * alpha * err * data[data_idx][0][w_idx] for (w_idx, val) in enumerate(self.w)]
            error = self.ms_error(data)

        return epochs



class AdalineBias:
    def __init__(self, size, f, max_init_weight = 0.2):
        self.max_init_weight = max_init_weight
        self.bias = 0
        self.f = f
        self.init_w(size)


    def init_w(self, size):
        self.w = []
        for i in range(size):
            value = 0
            value += (random.random() * 2 - 1) * self.max_init_weight
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
    
        # print("Learning completed")
        # print("Epochs: " + str(epochs))
        # print("Error: " + str(error))

        return epochs

def unipolar(value, theta = 0):
    return 1 if value >= theta else 0


def bipolar(value, theta = 0):
    return 1 if value >= theta else -1




def test_perceptron_theta(theta):
    model = Perceptron(2, unipolar, theta)
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    return model.learn(data, 0.1)
    # print("AND(1,1): " + str(model.predict([1,1])))
    # print("AND(0,1): " + str(model.predict([0,1])))
    # print("AND(0,0): " + str(model.predict([0,0])))
    # print("AND(1,0): " + str(model.predict([1,0])))


# Perceptron zad 1
def tests_1():
    # Nie jest w stanie wyuczyć dla theta 0 / 0.1
    print("\nTEST THETA\n")
    for i in range(9):
        theta = 0.1 * (i + 2)
        sum = 0
        for j in range(10):
            sum += test_perceptron_theta(theta)
        print(str(theta) + ', ' + str(sum/10))


def test_perceptron_init_w(range):
    model = PerceptronBias(2, unipolar, 0.2, range)
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    return model.learn(data, 0.1)


def test_adaline_init_w(range):
    model = AdalineBias(2, bipolar, range)
    data = [([-1, -1], -1), ([-1, 1], -1), ([1, -1], -1), ([1, 1], 1)]
    return model.learn(data, 0.1, 0.3)


# Perc 2 Ada 1
def tests_2():
    print("\nTEST INIT WEIGHTS \n")
    print("Perceptron")
    for i in range(10):
        weight_range = 0.1 * (i+1)
        sum = 0
        for j in range(10):
            sum += test_perceptron_init_w(weight_range)
        print(str(weight_range) + ', ' + str(sum/10))

    print("Adaline")
    for i in range(10):
        weight_range = 0.1 * (i+1)
        sum = 0
        for j in range(10):
            sum += test_adaline_init_w(weight_range)
        print(str(weight_range) + ', ' + str(sum/10))


def test_perceptron_alpha(alpha):
    model = PerceptronBias(2, unipolar, 0.2)
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    return model.learn(data, alpha)


def test_adaline_alpha(alpha):
    model = AdalineBias(2, bipolar)
    data = [([-1, -1], -1), ([-1, 1], -1), ([1, -1], -1), ([1, 1], 1)]
    return model.learn(data, alpha, 0.4)


# Perc 3, Ada 2
def tests_3():
    print("\nTEST ALPHA \n")
    print("Perceptron")
    alphas = [0.005, 0.01, 0.03, 0.05, 0.1, 0.15]
    for alpha in alphas:
        sum = 0
        for j in range(10):
            sum += test_perceptron_alpha(alpha)
        print(str(alpha) + ', ' + str(sum/10))

    print("Adaline")
    for alpha in alphas:
        sum = 0
        for j in range(10):
            sum += test_adaline_alpha(alpha)
        print(str(alpha) + ', ' + str(sum/10))


def test_perceptron_func(f, uni):
    model = PerceptronBias(2, f, 0.2)
    data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)] if uni else [([-1, -1], -1), ([-1, 1], -1), ([1, -1], -1), ([1, 1], 1)]
    return model.learn(data, 0.1)


# Perc 4
def tests_4():
    print("\nTEST ACT FUNC\n")
    print("Unipolar")
    sum = 0
    for i in range(100):
        sum += test_perceptron_func(unipolar, True)
    print(sum/100)
    print("Bipolar")
    sum = 0
    for i in range(100):
        sum += test_perceptron_func(bipolar, False)
    print(sum/100)


# xDxDxdxdDXD


if __name__ == "__main__":
    tests_1()
    tests_2()
    tests_3()
    tests_4()

    # model = PerceptronBias(2, unipolar, 0.2)
    # data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    # model.learn(data, 0.1)

    # model = AdalineBias(2, bipolar)
    # data = [([-1, -1], -1), ([-1, 1], -1), ([1, -1], -1), ([1, 1], 1)]
    # model.learn(data, 0.05, 0.3)
    # print("AND(0,1): " + str(model.predict([-1,1])))
    # print("AND(1,1): " + str(model.predict([1,1])))
    # print("AND(0,0): " + str(model.predict([-1,-1])))
    # print("AND(1,0): " + str(model.predict([1,-1])))

    # print("AND(1,1): " + str(model.predict_with_function([1,1])))
    # print("AND(0,1): " + str(model.predict_with_function([-1,1])))
    # print("AND(0,0): " + str(model.predict_with_function([-1,-1])))
    # print("AND(1,0): " + str(model.predict_with_function([1,-1])))
