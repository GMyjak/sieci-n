import numpy
from mnist import MNIST

MIN_ACT_SIGM = -200
MAX_ACT_SIGM = 200

NUM_OF_EXPERIMENTS = 10

GAMMA = 0.9

MOMENTUM = False
NESTEROV = True


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
        self.last_weight_delta = []
        self.last_bias_delta = []


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
        tr_data = numpy.array(train_data_samples) / 255
        tr_labels = numpy.array([label_matrix[i] for i in train_data_labels])
        vl_data = numpy.array(valid_data_samples) / 255
        vl_labels = numpy.array([label_matrix[i] for i in valid_data_labels])

        accs = []
        errs = []
        epochs = []

        last_validation_err = 1000

        acc = self.get_accuracy(vl_data, vl_labels)
        # print("INIT ACC:", acc)
        epoch_count = 0
        epochs_without_improvement = 0
        best_err = 9999999

        while epochs_without_improvement < 3 and epoch_count < 200 and last_validation_err > threshold:
            # single epoch
            epoch_count += 1
            for i in range(int(tr_data.shape[0]/batch_size)):
                start_index = i * batch_size
                end_index = max((i + 1) * batch_size - 1, tr_data.shape[0])
                batch = tr_data[start_index:end_index,:]
                batch_labels = tr_labels[start_index:end_index,:]
                start_err, end_err, diff = self.learn_minibatch(batch, batch_labels, alpha=alpha)

            acc = self.get_accuracy(vl_data, vl_labels)
            pred_after_update = self.batch_predict(vl_data)
            new_valid_err = nll(pred_after_update, vl_labels)
            if new_valid_err > last_validation_err:
                epochs_without_improvement += 1
            else:
                self.save_best_weights()
                epochs_without_improvement = 0
                best_err = new_valid_err
            last_validation_err = new_valid_err

            epochs.append(epoch_count)
            errs.append(end_err)
            accs.append(acc)

            print("EPOCH:", epoch_count)
            print("TRAIN ERROR:", end_err)
            print("VALID ERROR:", last_validation_err)
            print("ACC:", acc)

        if epochs_without_improvement > 0:
            self.load_best_weights()
            epoch_count -= 3
            acc = self.get_accuracy(vl_data, vl_labels)
            start_err, end_err, diff = self.learn_minibatch(batch, batch_labels, alpha=alpha)
            end_err = start_err

        # end report
        print("TOTAL EPOCH:", epoch_count)
        print("END TRAIN ERROR:", end_err)
        print("END VALID ERROR:", last_validation_err)
        print("END ACC:", acc)
        print("EARLY STOP:", epochs_without_improvement > 0)

        return epoch_count, end_err, acc, last_validation_err, epochs_without_improvement > 0, epochs, errs, accs



    # batch [i_sample, i_neuron]
    def learn_minibatch(self, batch, pred, alpha=0.001):
        
        # Propagacja wprzód
        out0 = self.layers[0].batch_output(batch)
        act0 = self.layers[0].batch_activated_output(batch)

        out1 = self.layers[1].batch_output(act0)
        act1 = self.layers[1].batch_activated_output(act0)

        #out2 = self.layers[2].batch_output(act1)
        #act2 = self.layers[2].batch_activated_output(act1)

        err_init = nll(act1, pred)

        # Liczenie gradientów
        delta_matrix_1 = numpy.zeros((batch.shape[0], self.layers[1].output_size))
        for i in range(batch.shape[0]):
            smax_der_res = softmax_der(act1[i,:], pred[i])
            delta_matrix_1[i,:] = smax_der_res

        der0 = numpy.vectorize(self.layers[0].function_der)
        delta_matrix_0 = (delta_matrix_1 @ self.layers[1].weights.T) * der0(out0)

        #der0 = numpy.vectorize(self.layers[0].function_der)
        #delta_matrix_0 = (delta_matrix_1 @ self.layers[1].weights.T) * der0(out0)
        
        # Aktualizacja wag
        weight_delta_1 = numpy.zeros(shape=self.layers[1].weights.shape)
        weight_delta_0 = numpy.zeros(shape=self.layers[0].weights.shape)
        bias_delta_1 = numpy.zeros(shape=self.layers[1].bias.shape)
        bias_delta_0 = numpy.zeros(shape=self.layers[0].bias.shape)
        for i in range(batch.shape[0]):
            weight_delta_1 += (alpha / batch.shape[0]) * numpy.outer(act0[i,:], delta_matrix_1[i,:])
            weight_delta_0 += (alpha / batch.shape[0]) * numpy.outer(batch[i,:], delta_matrix_0[i,:])

            bias_delta_1 += (alpha / batch.shape[0]) * delta_matrix_1[i,:]
            bias_delta_0 += (alpha / batch.shape[0]) * delta_matrix_0[i,:]

        if MOMENTUM:
            if self.layers[0].last_weight_delta != []:
                weight_delta_1 += GAMMA * self.layers[1].last_weight_delta
                weight_delta_0 += GAMMA * self.layers[0].last_weight_delta
                bias_delta_1 += GAMMA * self.layers[1].last_bias_delta
                bias_delta_0 += GAMMA * self.layers[0].last_bias_delta
            self.layers[1].last_weight_delta = weight_delta_1
            self.layers[0].last_weight_delta = weight_delta_0
            self.layers[1].last_bias_delta = bias_delta_1
            self.layers[0].last_bias_delta = bias_delta_0
        elif NESTEROV:
            weight_delta_1_tmp = self.layers[1].last_weight_delta
            weight_delta_0_tmp = self.layers[0].last_weight_delta
            bias_delta_1_tmp = self.layers[1].last_bias_delta
            bias_delta_0_tmp = self.layers[0].last_bias_delta
            if weight_delta_1_tmp != []:
                self.layers[1].last_weight_delta = GAMMA * self.layers[1].last_weight_delta - weight_delta_1
                self.layers[0].last_weight_delta = GAMMA * self.layers[0].last_weight_delta - weight_delta_0
                self.layers[1].last_bias_delta = GAMMA * self.layers[1].last_bias_delta - bias_delta_1
                self.layers[0].last_bias_delta = GAMMA * self.layers[0].last_bias_delta - bias_delta_0
            else:
                self.layers[1].last_weight_delta = weight_delta_1
                self.layers[0].last_weight_delta = weight_delta_0
                self.layers[1].last_bias_delta = bias_delta_1
                self.layers[0].last_bias_delta = bias_delta_0
            weight_delta_1 = -((1 + GAMMA) * weight_delta_1)
            weight_delta_0 = -((1 + GAMMA) * weight_delta_0)
            bias_delta_1 = -((1 + GAMMA) * bias_delta_1)
            bias_delta_0 = -((1 + GAMMA) * bias_delta_0)
            if weight_delta_1_tmp != []:
                weight_delta_1 += GAMMA * weight_delta_1_tmp
                weight_delta_0 += GAMMA * weight_delta_0_tmp
                bias_delta_1 += GAMMA * bias_delta_1_tmp
                bias_delta_0 += GAMMA * bias_delta_0_tmp
            else:
                print("XD")
                
        self.layers[1].weights = self.layers[1].weights - weight_delta_1
        self.layers[0].weights = self.layers[0].weights - weight_delta_0    
        self.layers[1].bias = self.layers[1].bias - bias_delta_1
        self.layers[0].bias = self.layers[0].bias - bias_delta_0

        pred_after_update = self.batch_predict(batch)
        err_after = nll(pred_after_update, pred)

        diff = err_init - err_after
        return (err_init, err_after, diff)


def relu(z):
    return z if z >= 0 else 0


def relu_der(z):
    if z == 0:
        return 0.5
    else: return 0 if z < 0 else 1


def sigmoid(z):
    #z = numpy.clip(z, -10, 10)
    return 1 / (1 + numpy.exp(-z))


def sigmoid_der(z):
    sigm = sigmoid(z)
    return sigm * (1 - sigm)


def tanh(z):
    #z = numpy.clip(z, -10, 10)
    return 2 / (1 + numpy.exp(-2 * z)) - 1


def tanh_der(z):
    th = tanh(z)
    return 1 - (th * th)


def softmax(results):
    #results = numpy.clip(results, -1000, 1000)
    results_e = [numpy.e ** a for a in results]
    sum = numpy.sum(results_e)
    return [ezj / sum for ezj in results_e]


def softmax_der(act, y):
    return numpy.array([-(y[i] - act[i]) for i in range(len(act))])


def nll(predicts, corrects):
    sum = 0
    for i in range(predicts.shape[0]):
        for j in range(predicts.shape[1]):
            sum += -numpy.log(max(predicts[i,j], 10**-5)) * corrects[i,j]
    return sum / predicts.shape[0]


def main():
    #perform_overnight_tests()
    learn_with_mnist()
    #test_hidden_layer_size()
    #learn_with_mnist_sigmoid()
    #compare_act_functions()

def test_hidden_layer_size():
    mndata = MNIST('./data/ubyte/')
    #todo


def perform_overnight_tests():
    mndata = MNIST('./data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()
    
    print("HIDDEN LAYER SIZE TESTS:")
    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 25, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(25, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.03)
        print("25 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 50, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(50, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.03)
        print("50 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.03)
        print("100 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 200, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(200, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.03)
        print("200 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 400, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(400, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.03)
        print("400 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)
    
    print("HL END\n")

    print("ALPHA TESTS:")
    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.001)
        print("0.001 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.005)
        print("0.005 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.01)
        print("0.01 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.05)
        print("0.05 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.1)
        print("0.1 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    print("ALPHA END\n")

    print("BATCH SIZE TESTS:")
    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 50, 0.05)
        print("50 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 100, 0.05)
        print("100 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 200, 0.05)
        print("200 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.05)
        print("400 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 1000, 0.05)
        print("1000 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    print("BATCH END\n")

    print("INIT WEIGHTS TESTS:")
    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.05)
        print("0.1 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.3))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.3))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.05)
        print("0.3 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.5))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.5))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.05)
        print("0.5 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model1 = MLP()
        model1.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.8))
        model1.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.8))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model1.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.05)
        print("0.8 TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    print("WEIGHTS END\n")


def compare_act_functions():
    mndata = MNIST('./data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    #for i in range(NUM_OF_EXPERIMENTS):
    #    model = MLP()
    #    model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
    #    model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
    #    (epoch_count, tr_err, acc, val_err, early, _, _, _) = model.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.1, 0.35)
    #    print("ReLU TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, sigmoid, sigmoid_der).init_weights(0.1))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        (epoch_count, tr_err, acc, val_err, early, _, _, _) = model.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.4, 0.35)
        print("Sigmoid TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)

    #for i in range(NUM_OF_EXPERIMENTS):
    #    model = MLP()
    #    model.add_layer(Layer(784, 100, tanh, tanh_der).init_weights(0.1))
    #    model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
    #    (epoch_count, tr_err, acc, val_err, early, _, _, _) = model.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.1, 0.35)
    #    print("Tanh TEST", i, "ACC:", acc, "T_ERR:", tr_err, "V_ERR:", val_err, "EPO:", epoch_count, "ES:", early)


def learn_with_mnist_sigmoid():
    mndata = MNIST('./data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    model = MLP()
    model.add_layer(Layer(784, 100, tanh, tanh_der).init_weights(0.1))
    model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
    model.learn(tr_images, tr_labels, vl_images, vl_labels, 400, 0.1)


def learn_with_mnist():
    mndata = MNIST('./data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    model = MLP()
    model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
    #model.add_layer(Layer(30, 30, relu, relu_der).init_weights(0.1))
    model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
    model.learn(tr_images[0:10000], tr_labels[0:10000], vl_images[0:1000], vl_labels[0:1000], 400, 0.1)


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
    