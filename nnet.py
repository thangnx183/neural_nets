import numpy as np
import random
import getdat

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(- z) )

class network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [ np.random.randn(y,1) for y in sizes[1:] ]
        self.weights = [ np.random.randn(i,j) for i,j in zip( sizes[1:] , sizes[:-1]) ]

    def forward(self, a):
        #print a.shape
        for w, b in zip(self.weights, self.biases):
            #print w
            a = sigmoid( np.dot(w,a) + b)
            #print sigmoid(np.dot(w,a) + b)
            #print a
            #print a.shape
        #print a
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data :
            n_test = len(test_data)

        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k + mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            #print len(mini_batches[49])
            for mini_batch in mini_batches:
                 self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0} : {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "epoch {0} complete ".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [ nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b) ]
            nabla_w = [ nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w) ]

        self.biases = [ b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b) ]
        self.weights = [ w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w) ]

    def backprop(self, x, y):
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]

        activation = x
        activations = [x]

        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        g = sigmoid(zs[-1])*(1 - sigmoid(zs[-1]))

        delta = (activations[-1] - y) * g

        nabla_w[-1] = delta * activations[-2].transpose()
        nabla_b[-1] = delta


        for l in xrange(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid(zs[-l]) * (1 - sigmoid(zs[-l]))
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b[-l] = delta

        return (nabla_b, nabla_w)

    def evaluate(self,test_data):
        test_result = [ ( np.argmax(self.forward(x) ), y ) for (x,y) in test_data]

        return sum( int( (x == 0 and y[0] == 1) or (x == 1 and y[0] == 0 )) for (x,y) in test_result )

nnet = network([400,50,50,2])
training_set, test_set = getdat.getdata()

nnet.SGD(training_set,200,30,7.75,test_set)
