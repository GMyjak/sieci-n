import numpy
from mnist import MNIST


# jak zrobić to zadanie
# uaktualnianie wag - dla każdej warstwy:
#   zapamiętać aktywację - to łatwe
#   policzyć tchibo
# żeby policzyć tchibo:
#   gradient softmax - hardcode?
#   inne gradienty :)

class Layer:
    def __init__(self, input_size, output_size, function, function_der, smax = False):
        self.input_size = input_size
        self.output_size = output_size
        self.function = function
        self.function_der = function_der
        self.smax = smax
        self.bias = []
        self.weights = []
        self.last_activation = []


    def init_weights(self, std_dev):
        self.weights = numpy.random.normal(0, std_dev, size=(self.input_size, self.output_size))
        self.bias = numpy.random.normal(0, std_dev, size=self.output_size)
        return self


    def output(self, input):
        return input @ self.weights + self.bias


    def activated_output(self, input):
        if self.smax:
            self.last_activation = self.function(self.output(input))
        else:
            self.last_activation = [self.function(z) for z in self.output(input)]
        return self.last_activation


    def nl_error(self, sample, correct):
        sum = 0
        for j in len(sample):
            sum += -numpy.log(self.last_activation[j]) * correct[j]
        return sum



class MLP:
    def __init__(self):
        self.input_size = 0
        self.output_size = 0
        self.layers = []
    
    
    def add_layer(self, layer):
        self.layers.append(layer)
        self.input_size = layer.input_size if self.layers == [] else self.input_size
        self.output_size = layer.output_size
        return self

    
    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.activated_output(output)
        return output


    def learn(self, sample):
        self.predict(sample)



def relu(z):
    return z if z >= 0 else 0


def relu_der(z):
    if z == 0:
        return 0.5
    else: return 0 if z < 0 else 1


def sigmoid(z):
    return 1 / (1 + numpy.e ** -z)


def sigmoid_der(z):
    sigm = sigmoid(z)
    return sigm * (1 - sigm)


def tanh(z):
    return 2 / (1 + numpy.e ** (-2 * z)) - 1


def tanh_der(z):
    th = tanh(z)
    return 1 - (th * th)


def softmax(results):
    results_e = [numpy.e ** a for a in results]
    for a in results:
        print(a)
    sum = numpy.sum(results_e)
    return [ezj / sum for ezj in results_e]


def softmax_der(results):
    smax = softmax(results)
    return [-()]


def main():
    test_data()
    #test_mlp()


def test_data():
    mndata = MNIST('./zad2/data/ubyte/')
    images, labels = mndata.load_training()

    model = MLP()
    model.add_layer(Layer(784, 300, relu, relu_der).init_weights(0.01))
    model.add_layer(Layer(300, 300, relu, relu_der).init_weights(0.01))
    model.add_layer(Layer(300, 10, softmax, None, True).init_weights(0.01))
    print("PREDICTION:", model.predict(images[0]))
    print("REAL:", labels[0])


def test_mlp():
    model = MLP()
    model.add_layer(Layer(3, 10, relu, relu_der).init_weights(1))
    model.add_layer(Layer(10, 15, relu, relu_der).init_weights(1))
    model.add_layer(Layer(15, 5, softmax, None, True).init_weights(1))
    print(model.predict([0.2, 0.6, 0.3]))


def test_layer():
    model = Layer(2, 3, relu, relu_der)
    model.init_weights(1.4)
    print(model.weights)
    print(model.bias)
    print(model.output([1, 1]))
    print(model.activated_output([1, 1]))
    print(softmax(model.activated_output([1, 1])))


def test_functions():
    print("RELU")
    print(relu(-1))
    print(relu(-0.8))
    print(relu(0))
    print(relu(1))
    print(relu(3))
    print("SIGMOID")
    print(sigmoid(-1))
    print(sigmoid(-0.8))
    print(sigmoid(0))
    print(sigmoid(1))
    print(sigmoid(3))
    print("TANH")
    print(tanh(-1))
    print(tanh(-0.8))
    print(tanh(0))
    print(tanh(1))
    print(tanh(3))


if __name__ == '__main__':
    main()
    