import numpy
from mnist import MNIST

class Layer:
    def __init__(self, input_size, output_size, function):
        self.input_size = input_size
        self.output_size = output_size
        self.function = function
        self.bias = []
        self.weights = []


    def init_weights(self, std_dev):
        self.weights = numpy.random.normal(0, std_dev, size=(self.output_size, self.input_size))
        self.bias = numpy.random.normal(0, std_dev, size=self.output_size)
        return self


    def output(self, input):
        return self.weights @ numpy.transpose(input) + self.bias


    def activated_output(self, input):
        return [self.function(z) for z in self.output(input)]


#    def softmax(self, input):
#        results = self.activated_output(input)
#        results_e = [numpy.e ** a for a in results]
#        sum = numpy.sum(results_e)
#        return [ezj / sum for ezj in results_e]


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
        return softmax(output)


def relu(z):
    return z if z >= 0 else 0


def sigmoid(z):
    return 1 / (1 + numpy.e ** -z)


def tanh(z):
    return 2 / (1 + numpy.e ** (-2 * z)) - 1


def softmax(results):
    results_e = [numpy.e ** a for a in results]
    sum = numpy.sum(results_e)
    return [ezj / sum for ezj in results_e]


def main():
    test_data()


def test_data():
    mndata = MNIST('./zad2/data/ubyte/')
    images, labels = mndata.load_training()

    model = MLP()
    model.add_layer(Layer(784, 300, tanh).init_weights(1))
    model.add_layer(Layer(300, 300, tanh).init_weights(1))
    model.add_layer(Layer(300, 10, tanh).init_weights(1))
    print("PREDICTION:", model.predict(images[0]))
    print("REAL:", labels[0])


def test_mlp():
    model = MLP()
    model.add_layer(Layer(3, 5, relu).init_weights(1))
    model.add_layer(Layer(5, 5, relu).init_weights(1))
    model.add_layer(Layer(5, 4, relu).init_weights(1))
    print(model.predict([0.2, 0.6, 0.3]))


def test_layer():
    model = Layer(2, 3, relu)
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
    