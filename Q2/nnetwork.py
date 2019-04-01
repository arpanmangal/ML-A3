"""
The class of Neural Network
"""
import numpy as np
from plot import make_confusion_matrix

class NNetwork:
    def __init__ (self, num_input, sizes, num_output, batch_size):
        self.sizes = sizes
        self.sizes.append (num_output)
        self.sizes.insert(0, num_input)
        self.batch_size = batch_size

        self.L = len(sizes) - 1

        # Make the Weight and Biases vectors for each layer
        # self.weights = [None] * self.L
        # self.biases = [None] * self.L
        # for l in range(1, self.L):
        #     self.weights[l] = np.random.random((sizes[l], sizes[l-1]))
        #     self.biases[l] = np.random.random((sizes[l],))

        self.biases = [np.random.randn(currSize, 1) for currSize in sizes[1:]]
        self.weights = [np.random.randn(currSize, prevSize) for currSize, prevSize in zip(sizes[1:], sizes[:-1])]
            
        # self.biases[0] = np.array((0, 0))
        # self.weights[0] = np.array((0, 0))
        # self.biases = np.array(self.biases)
        # self.weights = np.array(self.weights)
        # for b in self.biases:
        #     print (b)
        # for w in self.weights:
        #     print (w)


    def predict (self, X):
        def predict_single (x):
            return np.argmax(self.feedforward(x))
        return np.array(list(map(predict_single, X)))

    def feedforward (self, input):
        # Run the neural network on the input and reuturn the output
        a = input
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, a) + b
            a = self.sigmoid(z)

        return a


    def train (self, X, Y, epochs, eta):
        # Y = Y.reshape(len(Y), 1)
        # print (Y.shape)
        # print (Y)
        # print (X.shape)
        # print (X[0].shape)
        trainData = np.array(list(zip(X, Y)))
        print (trainData.shape)
        for e in range(epochs):
            np.random.shuffle(trainData)

            mini_batches = [
                trainData[k:k+self.batch_size]
                for k in range(0, len(trainData), self.batch_size)
            ]
            for mini_batch in mini_batches:
                self.MBGD (mini_batch, eta)

            accuracy = self.evaluate (X, Y)
            print ("Epoch: %d | Accuracy: %.2f" % (e, 100 * accuracy))
            # print (self.biases)
            # print (self.weights)

        X, Y = zip(*trainData)
        predictions = self.predict (X)
        make_confusion_matrix (Y, predictions)


    def MBGD (self, mini_batch, eta):
        """
        Mini-Batch Gradient Descent
        Update the weights and biases for this mini_batch
        """
        # X, Y = zip(*mini_batch)
        del_biases = [np.zeros(b.shape) for b in self.biases]
        del_weights = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # Do a feedforward and compute z and delta for each layer
            x = x.reshape(len(x), 1)
            zs = [] # Set of z's
            azs = [] # Set of a's
            a = x
            azs.append(a)
            for b, w in zip(self.biases, self.weights):
                # print (w.shape, a.shape, b.shape)
                z = np.matmul(w, a) + b
                print (z.shape, w.shape, a.shape, b.shape)
                zs.append(z)
                a = self.sigmoid(z)
                azs.append (a)
                # print (b.shape, z.shape, a.shape)

            print (zs)
            exit(0)

            # Do a backpropogate
            gradient_C = (a - y)
            delta = np.multiply (gradient_C, self.sigmoid_prime(zs[-1]))
            del_biases[-1] += delta
            del_weights[-1] += np.matmul (delta.reshape(len(delta), 1), azs[-2].reshape(1, len(azs[-2])))

            for l in range (2, self.L + 1):
                sp = self.sigmoid_prime(z[-1])
                delta = np.matmul (self.weights[-l + 1].transpose(), delta) * sp
                del_biases[-l] += delta
                del_weights[-l] += np.matmul (delta.reshape(len(delta), 1), azs[-l-1].reshape(1, len(azs[-l-1])))

            # for l in range (self.L - 2, -1, -1):
            #     delta = np.multiply( (np.matmul(np.transpose(self.weights[l+1]), deltas[l+1])), self.sigmoid_prime(z[l]))
            #     del_biases[l] += delta
            #     del_weights[l] += np.matmul (delta.reshape(len(delta), 1), azs[l].reshape(1, len(azs[l])))


        del_biases = [db / len(mini_batch) for db in del_biases]
        del_weights = [dw / len(mini_batch) for dw in del_weights]

        self.biases = [b - eta * db for b, db in zip(self.biases, del_biases)]
        self.weights = [w - eta * dw for w, dw in zip(self.weights, del_weights)]


    # Other Functions
    def evaluate (self, X, Y):
        predictions = self.predict (X)
        accuracy = self.accuracy (Y, predictions)
        return accuracy

    def accuracy (self, trueY, predictions):
        return np.sum(trueY == predictions) / len(trueY)

    # Helper Functions
    def sigmoid (self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime (self, z):
        sig = self.sigmoid (z)
        return np.multiply(sig, 1-sig)
