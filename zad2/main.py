import numpy
from mnist import MNIST

MIN_ACT_SIGM = -200
MAX_ACT_SIGM = 200

NUM_OF_EXPERIMENTS = 1

GAMMA = 0.9
BETA1 = 0.9
BETA2 = 0.9
EPS = 0.00000001

MOMENTUM = False
NESTEROV = False
ADAGRAD = False
ADADELTA = False
ADAM = False

NORMAL = True
XAVIER = False
HE = False

ALPHA = 0.01


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
        self.adagrad_w = []
        self.adagrad_b = None
        self.adam_m_w = []
        self.adam_m_b = None
        self.adam_v_w = []
        self.adam_v_b = None


    def init_weights(self, std_dev):
        if NORMAL:
            return self.init_weights_normal(std_dev)
        elif XAVIER:
            return self.xavier()
        elif HE:
            return self.he()

    
    def init_weights_normal(self, std_dev):
        self.weights = numpy.random.normal(0, std_dev, size=(self.input_size, self.output_size))
        self.bias = numpy.random.normal(0, std_dev, size=self.output_size)
        return self


    def xavier(self):
        var = 2 / (self.input_size + self.output_size)
        return self.init_weights_normal(numpy.sqrt(var))

    def he(self):
        var = 2 / (self.input_size)
        return self.init_weights_normal(numpy.sqrt(var))


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

        while epochs_without_improvement < 3 and epoch_count < 25 and last_validation_err > threshold:
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

            #print("EPOCH:", epoch_count)
            #print("TRAIN ERROR:", end_err)
            #print("VALID ERROR:", last_validation_err)
            #print("ACC:", acc)

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
            # w_del_tmp => v
            # mu => GAMMA
            weight_delta_1_tmp = self.layers[1].last_weight_delta
            weight_delta_0_tmp = self.layers[0].last_weight_delta
            bias_delta_1_tmp = self.layers[1].last_bias_delta
            bias_delta_0_tmp = self.layers[0].last_bias_delta
            if weight_delta_1_tmp != []:
                self.layers[1].last_weight_delta = -GAMMA * self.layers[1].last_weight_delta + weight_delta_1
                self.layers[0].last_weight_delta = -GAMMA * self.layers[0].last_weight_delta + weight_delta_0
                self.layers[1].last_bias_delta = -GAMMA * self.layers[1].last_bias_delta + bias_delta_1
                self.layers[0].last_bias_delta = -GAMMA * self.layers[0].last_bias_delta + bias_delta_0
            else:
                self.layers[1].last_weight_delta = weight_delta_1
                self.layers[0].last_weight_delta = weight_delta_0
                self.layers[1].last_bias_delta = bias_delta_1
                self.layers[0].last_bias_delta = bias_delta_0
            if weight_delta_1_tmp != []:
                weight_delta_1 = ((GAMMA + 2) * weight_delta_1) + GAMMA * weight_delta_1_tmp
                weight_delta_0 = ((GAMMA + 2) * weight_delta_0) + GAMMA * weight_delta_0_tmp
                bias_delta_1 = ((GAMMA + 2) * bias_delta_1) + GAMMA * bias_delta_1_tmp
                bias_delta_0 = ((GAMMA + 2) * bias_delta_0) + GAMMA * bias_delta_0_tmp
        elif ADAGRAD:
            weight_delta_1_tmp = weight_delta_1
            weight_delta_0_tmp = weight_delta_0
            bias_delta_1_tmp = bias_delta_1
            bias_delta_0_tmp = bias_delta_0

            if self.layers[1].adagrad_w != []:
                weight_delta_1 = ((alpha / numpy.sqrt(self.layers[1].adagrad_w + 0.00000001)) * weight_delta_1)
                weight_delta_0 = ((alpha / numpy.sqrt(self.layers[0].adagrad_w + 0.00000001)) * weight_delta_0)
                bias_delta_1 = ((alpha / numpy.sqrt(self.layers[1].adagrad_b + 0.00000001)) * bias_delta_1)
                bias_delta_0 = ((alpha / numpy.sqrt(self.layers[0].adagrad_b + 0.00000001)) * bias_delta_0)

                self.layers[1].adagrad_w += numpy.power(weight_delta_1_tmp, 2)
                self.layers[0].adagrad_w += numpy.power(weight_delta_0_tmp, 2)
                self.layers[1].adagrad_b += numpy.power(bias_delta_1_tmp, 2)
                self.layers[0].adagrad_b += numpy.power(bias_delta_0_tmp, 2)
                
            else:
                self.layers[1].adagrad_w = numpy.power(weight_delta_1_tmp, 2)
                self.layers[0].adagrad_w = numpy.power(weight_delta_0_tmp, 2)
                self.layers[1].adagrad_b = numpy.power(bias_delta_1_tmp, 2)
                self.layers[0].adagrad_b = numpy.power(bias_delta_0_tmp, 2)
        elif ADADELTA:
            new_rms_w_1 = numpy.power(weight_delta_1, 2)
            new_rms_w_0 = numpy.power(weight_delta_0, 2)
            new_rms_b_1 = numpy.power(bias_delta_1, 2)
            new_rms_b_0 = numpy.power(bias_delta_0, 2)

            if self.layers[1].adagrad_w != []:
                
                weight_delta_1 = (numpy.sqrt(self.layers[1].adagrad_w + EPS) / numpy.sqrt(new_rms_w_1 + EPS)) * weight_delta_1
                weight_delta_0 = (numpy.sqrt(self.layers[0].adagrad_w + EPS) / numpy.sqrt(new_rms_w_0 + EPS)) * weight_delta_0
                bias_delta_1 = (numpy.sqrt(self.layers[1].adagrad_b + EPS) / numpy.sqrt(new_rms_b_1 + EPS)) * bias_delta_1
                bias_delta_0 = (numpy.sqrt(self.layers[0].adagrad_b + EPS) / numpy.sqrt(new_rms_b_0 + EPS)) * bias_delta_0
                
            self.layers[1].adagrad_w = new_rms_w_1
            self.layers[0].adagrad_w = new_rms_w_0
            self.layers[1].adagrad_b = new_rms_b_1
            self.layers[0].adagrad_b = new_rms_b_0
        elif ADAM:
            if self.layers[1].adam_m_w != []:
                m_1_w = BETA1 * self.layers[1].adam_m_w + ((1 - BETA1) * weight_delta_1)
                m_0_w = BETA1 * self.layers[0].adam_m_w + ((1 - BETA1) * weight_delta_0)
                m_1_b = BETA1 * self.layers[1].adam_m_b + ((1 - BETA1) * bias_delta_1)
                m_0_b = BETA1 * self.layers[0].adam_m_b + ((1 - BETA1) * bias_delta_0)
                v_1_w = BETA2 * self.layers[1].adam_v_w + ((1 - BETA2) * numpy.power(weight_delta_1, 2))
                v_0_w = BETA2 * self.layers[0].adam_v_w + ((1 - BETA2) * numpy.power(weight_delta_0, 2))
                v_1_b = BETA2 * self.layers[1].adam_v_b + ((1 - BETA2) * numpy.power(bias_delta_1, 2))
                v_0_b = BETA2 * self.layers[0].adam_v_b + ((1 - BETA2) * numpy.power(bias_delta_0, 2))

            else:
                m_1_w = (1 - BETA1) * weight_delta_1
                m_0_w = (1 - BETA1) * weight_delta_0
                m_1_b = (1 - BETA1) * bias_delta_1
                m_0_b = (1 - BETA1) * bias_delta_0
                v_1_w = (1 - BETA2) * numpy.power(weight_delta_1, 2)
                v_0_w = (1 - BETA2) * numpy.power(weight_delta_0, 2)
                v_1_b = (1 - BETA2) * numpy.power(bias_delta_1, 2)
                v_0_b = (1 - BETA2) * numpy.power(bias_delta_0, 2)

            m_1_w_h = m_1_w / (1 - BETA1)
            m_0_w_h = m_0_w / (1 - BETA1)
            m_1_b_h = m_1_b / (1 - BETA1)
            m_0_b_h = m_0_b / (1 - BETA1)
            v_1_w_h = v_1_w / (1 - BETA2)
            v_0_w_h = v_0_w / (1 - BETA2)
            v_1_b_h = v_1_b / (1 - BETA2)
            v_0_b_h = v_0_b / (1 - BETA2)

            self.layers[1].adam_m_w = m_1_w
            self.layers[0].adam_m_w = m_0_w
            self.layers[1].adam_m_b = m_1_b
            self.layers[0].adam_m_b = m_0_b
            self.layers[1].adam_v_w = v_1_w
            self.layers[0].adam_v_w = v_0_w
            self.layers[1].adam_v_b = v_1_b
            self.layers[0].adam_v_b = v_0_b

            weight_delta_1 = (alpha / (numpy.sqrt(v_1_w_h) + EPS)) * m_1_w_h
            weight_delta_0 = (alpha / (numpy.sqrt(v_0_w_h) + EPS)) * m_0_w_h
            bias_delta_1 = (alpha / (numpy.sqrt(v_1_b_h) + EPS)) * m_1_b_h
            bias_delta_0 = (alpha / (numpy.sqrt(v_0_b_h) + EPS)) * m_0_b_h


                
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
    results = numpy.clip(results, -200, 200)
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
    tests_init()


def learn_with_mnist():
    mndata = MNIST('./zad2/data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    model = MLP()
    model.add_layer(Layer(784, 200, relu, relu_der).init_weights(0.1))
    model.add_layer(Layer(200, 10, softmax, softmax_der, True).init_weights(0.1))
    model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA)


def tests():
    global MOMENTUM
    global NESTEROV
    global ADAGRAD
    global ADADELTA
    global ADAM

    mndata = MNIST('./zad2/data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    #print("NORMAL")
    #
    #for i in range(NUM_OF_EXPERIMENTS):
    #    model = MLP()
    #    model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
    #    model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
    #    model.learn(tr_images[0:2000], tr_labels[0:2000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    MOMENTUM = True
    print("MOMENTUM")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    MOMENTUM = False
    NESTEROV = True
    print("NESTEROV")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    NESTEROV = False
    ADAGRAD = True
    print("ADAGRAD")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    ADAGRAD = False
    ADADELTA = True
    print("ADADELTA")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    ADADELTA = False
    ADAM = True
    print("ADAM")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.1))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.1))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

def tests_init():
    global ADAGRAD
    global NORMAL
    global HE
    global XAVIER
    ADAGRAD = True

    mndata = MNIST('./zad2/data/ubyte/')
    tr_images, tr_labels = mndata.load_training()
    vl_images, vl_labels = mndata.load_testing()

    print("normal + relu")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.5))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.5))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    print("normal + sigmoid")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, sigmoid, sigmoid_der).init_weights(0.5))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.5))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    NORMAL = False
    XAVIER = True

    print("xavier + relu")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.5))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.5))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    print("xavier + sigmoid")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, sigmoid, sigmoid_der).init_weights(0.5))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.5))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    XAVIER = False
    HE = True

    print("he + relu")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, relu, relu_der).init_weights(0.5))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.5))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    print("he + sigmoid")

    for i in range(NUM_OF_EXPERIMENTS):
        model = MLP()
        model.add_layer(Layer(784, 100, sigmoid, sigmoid_der).init_weights(0.5))
        model.add_layer(Layer(100, 10, softmax, softmax_der, True).init_weights(0.5))
        model.learn(tr_images[0:5000], tr_labels[0:5000], vl_images[0:2000], vl_labels[0:2000], 200, ALPHA, threshold=0.35)

    

if __name__ == '__main__':
    main()
    