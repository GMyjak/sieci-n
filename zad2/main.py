import numpy
from mnist import MNIST
import matplotlib.pyplot as plt

MIN_ACT_SIGM = -10
MAX_ACT_SIGM = 10

NUM_OF_EXPERIMENTS = 5


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


class Layer:
    def __init__(self, input_size, output_size, function, function_der, smax = False):
        self.input_size = input_size
        self.output_size = output_size
        self.function = function
        self.function_der = function_der
        self.smax = smax
        self.bias = []
        self.weights = []
        self.best_bias = None
        self.best_weights = None


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


    def normalize_weights(self, min, max):
        self.weights = numpy.clip(self.weights, min, max)
        self.bias = numpy.clip(self.bias, min, max)

    
    def save_best_weights(self):
        self.best_bias = self.bias
        self.best_weights = self.weights


    def load_best_weights(self):
        self.bias = self.best_bias
        self.weights = self.best_weights


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

    
    def batch_predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.batch_activated_output(outputs)
        return outputs

    
    def normalize_weights(self, min, max):
        for layer in self.layers:
            layer.normalize_weights(min, max)

    
    def save_best_weights(self):
        for layer in self.layers:
            layer.save_best_weights()


    def load_best_weights(self):
        for layer in self.layers:
            layer.load_best_weights()

    
    def get_accuracy(self, data, labels):
        outputs = self.batch_predict(data)
        predicts = [numpy.argmax(row) for row in outputs]
        correct_predicts = [numpy.argmax(row) for row in labels]
        sum = 0
        for i in range(data.shape[0]):
            if predicts[i] == correct_predicts[i]:
                sum += 1
        acc = sum / data.shape[0]
        return int(acc * 1000) / 10


    def learn(self, train_data_samples, train_data_labels, valid_data_samples, valid_data_labels, batch_size, alpha, threshold=-1):
        tr_data = numpy.array(train_data_samples)
        tr_labels = numpy.array([label_matrix[i] for i in train_data_labels])
        vl_data = numpy.array(valid_data_samples)
        vl_labels = numpy.array([label_matrix[i] for i in valid_data_labels])
    
        accs = []
        errs = []
        epochs = []

        acc = self.get_accuracy(vl_data, vl_labels)
        # print("INIT ACC:", acc)
        epoch_count = 0
        epochs_without_improvement = 0
        best_err = 9999999

        while epochs_without_improvement < 3 and epoch_count < 300 and best_err > threshold:
            # single epoch
            epoch_count += 1
            for i in range(int(tr_data.shape[0]/batch_size)):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size - 1
                batch = tr_data[start_index:end_index,:]
                batch_labels = tr_labels[start_index:end_index,:]
                start_err, end_err, diff = self.learn_minibatch(batch, batch_labels, alpha=alpha)
            acc = self.get_accuracy(vl_data, vl_labels)
            if end_err > best_err:
                epochs_without_improvement += 1
            else:
                self.save_best_weights()
                epochs_without_improvement = 0
                best_err = end_err

            epochs.append(epoch_count)
            errs.append(end_err)
            accs.append(acc)

            #print("EPOCH:", epoch_count)
            #print("ERROR:", end_err)
            #print("ACC:", acc)

        if epochs_without_improvement > 0:
            self.load_best_weights()
            epoch_count -= 3
            acc = self.get_accuracy(vl_data, vl_labels)
            start_err, end_err, diff = self.learn_minibatch(batch, batch_labels, alpha=alpha)
            end_err = start_err

        # end report
        print("TOTAL EPOCH:", epoch_count)
        print("END ERROR:", end_err)
        print("END ACC:", acc)
        print("EARLY STOP:", epochs_without_improvement > 0)

        return epoch_count, end_err, acc, epochs_without_improvement > 0, epochs, errs, accs



    # batch [i_sample, i_neuron]
    def learn_minibatch(self, batch, pred, alpha=0.001):

        # Propagacja wprzód
        out0 = self.layers[0].batch_output(batch)
        act0 = self.layers[0].batch_activated_output(batch)

        out1 = self.layers[1].batch_output(act0)
        act1 = self.layers[1].batch_activated_output(act0)

        out2 = self.layers[2].batch_output(act1)
        act2 = self.layers[2].batch_activated_output(act1)

        err_init = nll(act2, pred)

        # Liczenie gradientów
        delta_matrix_2 = numpy.zeros((batch.shape[0], self.layers[2].output_size))
        for i in range(batch.shape[0]):
            smax_der_res = softmax_der(act2[i,:], pred[i])
            delta_matrix_2[i,:] = smax_der_res

        der1 = numpy.vectorize(self.layers[1].function_der)
        delta_matrix_1 = (delta_matrix_2 @ self.layers[2].weights.T) * der1(out1)

        der0 = numpy.vectorize(self.layers[0].function_der)
        delta_matrix_0 = (delta_matrix_1 @ self.layers[1].weights.T) * der0(out0)
        
        # Aktualizacja wag
        for i in range(batch.shape[0]):
            self.layers[2].weights = self.layers[2].weights - (alpha / batch.shape[0]) * numpy.outer(act1[i,:], delta_matrix_2[i,:])
            self.layers[1].weights = self.layers[1].weights - (alpha / batch.shape[0]) * numpy.outer(act0[i,:], delta_matrix_1[i,:])
            self.layers[0].weights = self.layers[0].weights - (alpha / batch.shape[0]) * numpy.outer(batch[i,:], delta_matrix_0[i,:])
            self.layers[2].bias = self.layers[2].bias - (alpha / batch.shape[0]) * delta_matrix_2[i,:]
            self.layers[1].bias = self.layers[1].bias - (alpha / batch.shape[0]) * delta_matrix_1[i,:]
            self.layers[0].bias = self.layers[0].bias - (alpha / batch.shape[0]) * delta_matrix_0[i,:]

        self.normalize_weights(MIN_ACT_SIGM, MAX_ACT_SIGM)

        pred_after_update = self.batch_predict(batch)
        err_after = nll(pred_after_update, pred)

        diff = err_init - err_after
        return (err_init, err_after, diff)


def relu(z):
    return max(z, 100) if z >= 0 else 0


def relu_der(z):
    if z == 0:
        return 0.5
    else: return 0 if z < 0 else 1


def sigmoid(z):
    z = numpy.clip(z, MIN_ACT_SIGM, MAX_ACT_SIGM)
    return 1 / (1 + numpy.e ** -z)


def sigmoid_der(z):
    sigm = sigmoid(z)
    return sigm * (1 - sigm)


def tanh(z):
    z = numpy.clip(z, MIN_ACT_SIGM, MAX_ACT_SIGM)
    return 2 / (1 + numpy.e ** (-2 * z)) - 1


def tanh_der(z):
    th = tanh(z)
    return 1 - (th * th)


def softmax(results):
    results_e = [numpy.e ** (a/numpy.max(results)) for a in results]
    sum = numpy.sum(results_e)
    return [ezj / sum for ezj in results_e]


def softmax_der(act, y):
    return numpy.array([-(y[i] - act[i]) for i in range(len(act))])


def nll(predicts, corrects):
    sum = 0
    for i in range(predicts.shape[0]):
        for j in range(predicts.shape[1]):
            sum += -numpy.log(predicts[i,j]) * corrects[i,j]
    return sum / predicts.shape[0]


def main():
    test_hidden_layer_size()


def test_hidden_layer_size():
    mndata = MNIST('./data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    print("Start learning...")
    
    model1 = MLP()
    model1.add_layer(Layer(784, 30, relu, relu_der).init_weights(0.1))
    model1.add_layer(Layer(30, 30, relu, relu_der).init_weights(0.1))
    model1.add_layer(Layer(30, 10, softmax, softmax_der, True).init_weights(0.1))
    data1 = model1.learn(tr_images, tr_labels, vl_images, vl_labels, 1000, 0.01, threshold=0.25)

    print("Model 1 complete...")

    model2 = MLP()
    model2.add_layer(Layer(784, 50, relu, relu_der).init_weights(0.1))
    model2.add_layer(Layer(50, 50, relu, relu_der).init_weights(0.1))
    model2.add_layer(Layer(50, 10, softmax, softmax_der, True).init_weights(0.1))
    data2 = model2.learn(tr_images, tr_labels, vl_images, vl_labels, 1000, 0.01, threshold=0.30)

    print("Model 2 complete...")

    model3 = MLP()
    model3.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
    model3.add_layer(Layer(100, 100, relu, relu_der).init_weights(0.1))
    model3.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
    data3 = model3.learn(tr_images, tr_labels, vl_images, vl_labels, 1000, 0.01, threshold=0.30)

    print("Model 3 complete...")

    model4 = MLP()
    model4.add_layer(Layer(784, 300, relu, relu_der).init_weights(0.1))
    model4.add_layer(Layer(300, 100, relu, relu_der).init_weights(0.1))
    model4.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
    data4 = model4.learn(tr_images, tr_labels, vl_images, vl_labels, 1000, 0.01, threshold=0.30)

    print("Model 4 complete...")

    model5 = MLP()
    model5.add_layer(Layer(784, 500, relu, relu_der).init_weights(0.1))
    model5.add_layer(Layer(500, 150, relu, relu_der).init_weights(0.1))
    model5.add_layer(Layer(150, 10, softmax, softmax_der, True).init_weights(0.1))
    data5 = model5.learn(tr_images, tr_labels, vl_images, vl_labels, 1000, 0.01, threshold=0.30)

    print("Model 5 complete...")

    print(data1)
    print(data2)
    print(data3)
    print(data4)
    print(data5)

    x = range(100)
    epo1 = data1[4][0:99]
    err1 = data1[5][0:99]
    acc1 = data1[6][0:99]

    epo2 = data2[4][0:99]
    err2 = data2[5][0:99]
    acc2 = data2[6][0:99]

    epo3 = data3[4][0:99]
    err3 = data3[5][0:99]
    acc3 = data3[6][0:99]

    epo4 = data4[4][0:99]
    err4 = data4[5][0:99]
    acc4 = data4[6][0:99]

    epo5 = data5[4][0:99]
    err5 = data5[5][0:99]
    acc5 = data5[6][0:99]
    
    plt.plot(x, err1, label = '30, 30')
    plt.plot(x, err2, label = '50, 50')
    plt.plot(x, err3, label = '100, 100')
    plt.plot(x, err4, label = '300, 100')
    plt.plot(x, err5, label = '500, 150')
    
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.title('Spadek błędu w zależności od rozmiaru warstw ukrytych')
    plt.savefig('image1.png')
    plt.show()

    plt.plot(x, acc1, label = '30, 30')
    plt.plot(x, acc2, label = '50, 50')
    plt.plot(x, acc3, label = '100, 100')
    plt.plot(x, acc4, label = '300, 100')
    plt.plot(x, acc5, label = '500, 150')
    
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Wzrost skuteczności w zależności od rozmiaru warstw ukrytych')
    plt.savefig('image2.png')
    plt.show()


def learn_with_mnist():
    mndata = MNIST('./data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    model = MLP()
    model.add_layer(Layer(784, 30, relu, relu_der).init_weights(0.1))
    model.add_layer(Layer(30, 30, relu, relu_der).init_weights(0.1))
    model.add_layer(Layer(30, 10, softmax, softmax_der, True).init_weights(0.1))
    model.learn(tr_images, tr_labels, vl_images, vl_labels, 1000, 0.005)


def test_learn():
    model = MLP()
    model.add_layer(Layer(4, 3, relu, relu_der).init_weights(1))
    model.add_layer(Layer(3, 3, relu, relu_der).init_weights(1))
    model.add_layer(Layer(3, 2, softmax, softmax_der, True).init_weights(1))

    data = numpy.array([[0.1, 0.2, 0.5, 0.2],[0.5, 0.2, 0.4, 0.1],[0.4, 0.2, 0.9, 0.1]])
    pred = numpy.array([[1, 1], [1, 0], [0, 1]])
    init, after, diff = model.learn_minibatch(data, pred, 0.01)
    print("init:" , init)
    while diff > 0:
        init, after, diff = model.learn_minibatch(data, pred, 0.01)
        print(diff)
    print("end:" , after)


def test_data():
    mndata = MNIST('C:/Users/Grzegorz/source/repos/sieci-n/zad2/data/ubyte/')
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
    