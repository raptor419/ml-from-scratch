import numpy as np
import struct

def load_mnist():
    """Load MNIST data"""
    with open('train-labels-idx1-ubyte', 'rb') as f:
        struct.unpack(">II", f.read(8))
        y_train = np.fromfile(f, dtype=np.int8)

    with open('train-images-idx3-ubyte', 'rb') as f:
        _, _, rows, cols = struct.unpack(">IIII", f.read(16))
        X_train = np.fromfile(f, dtype=np.uint8).reshape(len(y_train), rows, cols)

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        struct.unpack(">II", f.read(8))
        y_test = np.fromfile(f, dtype=np.int8)

    with open('t10k-images-idx3-ubyte', 'rb') as f:
        _, _, rows, cols = struct.unpack(">IIII", f.read(16))
        X_test = np.fromfile(f, dtype=np.uint8).reshape(len(y_test), rows, cols)

    # flatten
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    # convert to one-hot encoding
    one_hot = np.zeros((len(y_train), 10))
    one_hot[np.arange(len(y_train)), y_train] = 1
    y_train = one_hot

    one_hot = np.zeros((len(y_test), 10))
    one_hot[np.arange(len(y_test)), y_test] = 1
    y_test = one_hot

    return X_train, y_train, X_test, y_test

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def d_cost(y, pred):
    return pred - y

class MultilayerPerceptron(object):
    """Implementation of a MultilayerPerceptron (MLP)
    Note: this code is NOT optimized and may run slowly!
    """
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.activation_fn = sigmoid
        self.d_activation_fn = d_sigmoid

        # initialize weights and biases to small, Gaussian values/noise (adjust for variance)
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def feedforward(self, X):
        """Feed input through this network
        
        Arguments:
            X {N x layers[0] array} -- Inputs where N is number of examples
        
        Returns:
            N x layers[-1] array -- returns activations of last layer
        """
        # pre-allocate activations and compute for each input
        activations = np.zeros((len(X), self.layers[-1]))
        for i, x in enumerate(X):
            a = x.reshape(-1, 1)
            for W, b in zip(self.weights, self.biases):
                a = self.activation_fn(W.dot(a) + b)
            activations[i] = a.reshape(-1)
        return activations

    def accuracy(self, y, pred):
        """Computes accuracy
        
        Arguments:
            y {N x layers[-1] array} -- ground-truth values
            pred {N x layers[-1] array} -- last layer activations from neural network
        
        Returns:
            float -- accuracy
        """
        return np.sum(np.argmax(y, axis=1) == np.argmax(pred, axis=1)) / len(y)

    def fit(self, X, y, epochs=10, batch_size=128, lr=1e-4):
        """Run stochastic gradient descent (SGD) on training data
        
        Arguments:
            X {N x layers[0] array} -- All training data
            y {N x layers[-1] array} -- All training labels
        
        Keyword Arguments:
            epochs {number} -- Number of epochs to train (default: {10})
            batch_size {number} -- Size of minibatch for SGD (default: {128})
            lr {number} -- learning rate (default: {1e-4})
        """
        N = len(y)
        for e in range(epochs):
            # shuffle and batch
            idx = np.random.permutation(N)
            shuffled_X, shuffled_y = X[idx], y[idx]
            batches = [zip(shuffled_X[b:b+batch_size], shuffled_y[b:b+batch_size]) for b in range(0, N, batch_size)]

            for i in range(len(batches)-1):
                X_train_batch, y_train_batch = zip(*batches[i])
                self._update_params(X_train_batch, y_train_batch, lr)

            # validate
            X_val_batch, y_val_batch = zip(*batches[-1])
            y_val_batch = np.array(y_val_batch)
            pred_val_batch = self.feedforward(X_val_batch)
            val_accuracy = self.accuracy(y_val_batch, pred_val_batch)
            print("Epoch {0}: Validation Accuracy: {1:.2f}".format(e+1, val_accuracy))

    def _update_params(self, X_train_batch, y_train_batch, lr):
        nabla_W = [np.zeros_like(W) for W in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        for x, y in zip(X_train_batch, y_train_batch): 
            delta_nabla_W, delta_nabla_b = self._backprop(x, y)
            nabla_W = [dnW + nW for nW, dnW in zip(delta_nabla_W, nabla_W)]
            nabla_b = [dnb + nb for nb, dnb in zip(delta_nabla_b, nabla_b)]

        self.weights = [W - (lr * nW / len(X_train_batch)) for W, nW in zip(self.weights, nabla_W)]
        self.biases = [b - (lr * nb / len(X_train_batch)) for b, nb in zip(self.biases, nabla_b)]

    def _backprop(self, x, y):
        nabla_W = [np.zeros(W.shape) for W in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # forward pass
        a = x.reshape(-1, 1)
        activations = [a]
        zs = []
        for W, b in zip(self.weights, self.biases):
            z = W.dot(a) + b
            zs.append(z)
            a = self.activation_fn(z)            
            activations.append(a)

        # backward pass
        nabla_C = d_cost(y.reshape(-1, 1), activations[-1])
        delta = np.multiply(nabla_C, self.d_activation_fn(zs[-1]))
        nabla_b[-1] = delta
        nabla_W[-1] = delta.dot(activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.multiply(self.weights[-l+1].T.dot(delta), self.d_activation_fn(z))
            nabla_b[-l] = delta
            nabla_W[-l] = delta.dot(activations[-l-1].T)
        return (nabla_W, nabla_b)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist()

    mlp = MultilayerPerceptron(layers=[784, 512, 10])
    mlp.fit(X_train, y_train)

    pred = mlp.feedforward(X_test)
    accuracy = mlp.accuracy(y_test, pred)
    print("Test accuracy: {0:.2f}".format(accuracy))
