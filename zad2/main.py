import numpy
from mnist import MNIST


class Layer:
    def __init__(self, input_size, output_size, function, function_der, smax = False):
        self.input_size = input_size
        self.output_size = output_size
        self.function = function
        self.function_der = function_der
        self.smax = smax
        self.bias = []
        self.weights = []
        self.last_batch_output = []
        self.last_batch_activation = []


    def init_weights(self, std_dev):
        self.weights = numpy.random.normal(0, std_dev, size=(self.input_size, self.output_size))
        self.bias = numpy.random.normal(0, std_dev, size=self.output_size)
        return self


    def output(self, input):
        return input @ self.weights + self.bias

    
    def batch_output(self, inputs):
        return inputs @ self.weights + self.bias


    def activated_output(self, input):
        if self.smax:
            return self.function(self.output(input))
        else:
            return [self.function(z) for z in self.output(input)]

    
    def batch_activated_output(self, inputs):
        bout = self.batch_output(inputs)
        if self.smax:
            return numpy.array([self.function(bout[i,:]) for i in range(bout.shape[0])])
        else:
            f = numpy.vectorize(self.function)
            return f(bout)


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

    # batch [i_sample, i_neuron]
    def learn_batch(self, batch, pred, alpha=0.001):

        # Propagacja wprzód
        out0 = self.layers[0].batch_output(batch)
        act0 = self.layers[0].batch_activated_output(batch)

        out1 = self.layers[1].batch_output(act0)
        act1 = self.layers[1].batch_activated_output(act0)

        out2 = self.layers[2].batch_output(act1)
        act2 = self.layers[2].batch_activated_output(act1)

        # Liczenie błędów

        # tchibo [i_sample, i_neuron, i_weight]
        tchibo2 = numpy.zeros((batch.shape[0], self.layers[2].output_size, self.layers[2].input_size))
        for i in range(batch.shape[0]):
            smax_der_res = softmax_der(out2[i,:], pred[i])
            err = numpy.outer(smax_der_res, act1[i,:])
            tchibo2[i,:,:] = err

        #print(tchibo2)

        tchibo1 = numpy.zeros((batch.shape[0], self.layers[1].output_size, self.layers[1].input_size))
        for i in range(batch.shape[0]):    
            z3a2 = self.layers[2].weights @ tchibo2[i,:,:]
            #print(z3a2)
            der1 = numpy.vectorize(self.layers[1].function_der)
            a2z2 = der1(out1[i,:])
            #print(a2z2)
            err = z3a2 @ a2z2
            err = numpy.outer(err, act0[i,:])
            #print(err)
            tchibo1[i,:,:] = err
        
        #print(tchibo1)

        tchibo0 = numpy.zeros((batch.shape[0], self.layers[0].output_size, self.layers[0].input_size))
        for i in range(batch.shape[0]):
            z2a1 = self.layers[1].weights @ tchibo1[i,:,:]
            #print(z2a1)
            der0 = numpy.vectorize(self.layers[0].function_der)
            a1z1 = der0(out0[i,:])
            #print(a1z1)
            err = z2a1 @ a1z1
            err = numpy.outer(err, batch[i,:])
            #print(err)
            tchibo0[i,:,:] = err
        
        # print(tchibo1)

        # Aktualizacja wag
        for i in range(batch.shape[0]):
            # print(tchibo2.shape)
            # print(act1.shape)
            # print(self.layers[2].weights.shape)
            do_akt = tchibo2 @ act1
            print(do_akt.shape)
    
            



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
    sum = numpy.sum(results_e)
    return [ezj / sum for ezj in results_e]


def softmax_der(z, y):
    smax = softmax(z)
    return [-(y[i] - smax[i]) for i in range(len(z))]


def main():
    test_learn()


def test_learn():
    model = MLP()
    model.add_layer(Layer(4, 3, relu, relu_der).init_weights(1))
    model.add_layer(Layer(3, 3, relu, relu_der).init_weights(1))
    model.add_layer(Layer(3, 2, softmax, softmax_der, True).init_weights(1))

    data = numpy.array([[0.1, 0.2, 0.5, 0.2],[0.5, 0.2, 0.4, 0.1],[0.4, 0.2, 0.9, 0.1]])
    pred = numpy.array([[1, 1], [1, 0], [0, 1]])
    model.learn_batch(data, pred)


def test_data():
    mndata = MNIST('./zad2/data/ubyte/')
    images, labels = mndata.load_training()

    model = MLP()
    model.add_layer(Layer(784, 300, relu, relu_der).init_weights(0.01))
    model.add_layer(Layer(300, 300, relu, relu_der).init_weights(0.01))
    model.add_layer(Layer(300, 10, softmax, softmax_der, True).init_weights(0.01))
    print("PREDICTION:", model.predict(images[0]))
    print("REAL:", labels[0])


def test_mlp():
    model = MLP()
    model.add_layer(Layer(3, 10, relu, relu_der).init_weights(1))
    model.add_layer(Layer(10, 15, relu, relu_der).init_weights(1))
    model.add_layer(Layer(15, 5, softmax, softmax_der, True).init_weights(1))
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
    