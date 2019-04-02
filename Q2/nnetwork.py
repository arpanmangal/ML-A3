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
        self.biases = [np.random.randn(currSize, 1) for currSize in sizes[1:]]
        self.weights = [np.random.randn(currSize, prevSize) for currSize, prevSize in zip(sizes[1:], sizes[:-1])]
            

    def predict (self, data):
        def predict_single (x):
            return np.argmax(self.feedforward(x))

        return np.array([predict_single(x) for x, y in data])

    def feedforward (self, a):
        # Run the neural network on the input and reuturn the output
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, a) + b
            a = self.sigmoid(z)
        return a


    def train (self, trainData, epochs, eta):
        accuracies = []
        losses = []

        Y = self.getTrueY(trainData)
        accuracy = self.evaluate (trainData, Y) * 100
        accuracies.append(accuracy)
        loss = self.loss(trainData)
        losses.append(loss)
        print ("Epoch: %d | Accuracy: %.2f | Loss: %.2f" % (0, accuracy, loss))

        for e in range(epochs):
            np.random.shuffle(trainData)
            Y = self.getTrueY(trainData)

            mini_batches = [
                trainData[k:k+self.batch_size]
                for k in range(0, len(trainData), self.batch_size)
            ]
            for mini_batch in mini_batches:
                self.MBGD (mini_batch, eta)

            accuracy = self.evaluate (trainData, Y) * 100
            loss = self.loss(trainData)
            accuracies.append(accuracy)
            losses.append(loss)
            print ("Epoch: %d | Accuracy: %.2f | Loss: %.2f" % (e+1, accuracy, loss))
        
        return accuracies, losses


    def MBGD (self, mini_batch, eta):
        """
        Mini-Batch Gradient Descent
        Update the weights and biases for this mini_batch
        """
        # X, Y = zip(*mini_batch)
        del_biases = [np.zeros(b.shape) for b in self.biases]
        del_weights = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            zs = [] # Set of z's
            azs = [] # Set of a's
            a = x
            azs.append(a)
            for b, w in zip(self.biases, self.weights):
                z = np.matmul(w, a) + b
                zs.append(z)
                a = self.sigmoid(z)
                azs.append (a)

            # Do a backpropogate
            gradient_C = (a - y)
            delta = np.multiply (gradient_C, self.sigmoid_prime(zs[-1]))
            del_biases[-1] += delta
            del_weights[-1] += np.matmul (delta, azs[-2].transpose())

            for l in range (2, self.L + 1):
                sp = self.sigmoid_prime(zs[-l])
                delta = np.matmul (self.weights[-l + 1].transpose(), delta) * sp
                del_biases[-l] += delta
                del_weights[-l] += np.matmul ( delta, azs[-l-1].transpose() )

        del_biases = [db / len(mini_batch) for db in del_biases]
        del_weights = [dw / len(mini_batch) for dw in del_weights]

        self.biases = [b - eta * db for b, db in zip(self.biases, del_biases)]
        self.weights = [w - eta * dw for w, dw in zip(self.weights, del_weights)]


    # Other Functions
    def evaluate (self, data, Y):
        predictions = self.predict (data)
        accuracy = self.accuracy (Y, predictions)
        return accuracy

    def getTrueY (self, data):
        return np.array([np.argmax(y) for x, y in data])

    def accuracy (self, trueY, predictions):
        return np.sum(trueY == predictions) / len(trueY)

    def loss (self, data):
        loss = 0
        for x, y in data:
            loss += np.sum(np.square(self.feedforward(x) - y))
        return loss / (2 * len(data))

    # Helper Functions
    def sigmoid (self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime (self, z):
        sig = self.sigmoid (z)
        sp = np.multiply(sig, 1-sig)
        return sp
