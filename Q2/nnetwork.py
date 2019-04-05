"""
The class of Neural Network
"""
import numpy as np
from plot import make_confusion_matrix

class NNetwork:
    def __init__ (self, num_input, sizes, num_output, batch_size, useRELU=False):
        self.sizes = sizes[:]
        self.sizes.append (num_output)
        self.sizes.insert(0, num_input)
        self.batch_size = batch_size
        self.useRELU = useRELU

        self.L = len(self.sizes) - 1

        # Make the Weight and Biases vectors for each layer
        self.biases = [np.random.randn(currSize, 1) for currSize in self.sizes[1:]]
        self.weights = [np.random.randn(currSize, prevSize) for currSize, prevSize in zip(self.sizes[1:], self.sizes[:-1])]
        
        np.random.seed(0)

    def predict (self, data):
        def predict_single (x):
            return np.argmax(self.feedforward(x))

        return np.array([predict_single(x) for x, y in data])

    def feedforward (self, a):
        # Run the neural network on the input and reuturn the output
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.matmul(w, a) + b
            a = self.sigmoid(z, False)
        for b, w in zip(self.biases[-1:], self.weights[-1:]):
            z = np.matmul(w, a) + b
            a = self.sigmoid(z, True)
        return a


    def train (self, trainData, eta, silent=False, max_epochs=1000, adaptive_eta=False):
        accuracies = []
        losses = []

        Y = self.getTrueY(trainData)
        accuracy = self.evaluate (trainData, Y) * 100
        accuracies.append(accuracy)
        loss = self.loss(trainData)
        losses.append(loss)
        if (not silent):
            print ("Epoch: %d | Accuracy: %.3f | Loss: %.6f" % (0, accuracy, loss))

        for e in range(max_epochs):
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
            if (not silent):
                print ("Epoch: %d | Accuracy: %.3f | Loss: %.6f | Eta: %.3f" % (e+1, accuracy, loss, eta))

            # Convergence / Divergence Criterion
            if (abs(losses[-1] - losses[-2]) <= 1e-5) or (losses[-1] - losses[-2] >= 0.1):
                break

            if adaptive_eta:
                loss_increase = losses[-1] - losses[-2]
                if ( loss_increase >= 0 and loss_increase < 1e-4 ):
                    eta /= 5
            
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
            for b, w in zip(self.biases[:-1], self.weights[:-1]):
                z = np.matmul(w, a) + b
                zs.append(z)
                a = self.sigmoid(z, False)
                azs.append (a)
            for b, w in zip(self.biases[-1:], self.weights[-1:]):
                z = np.matmul(w, a) + b
                zs.append(z)
                a = self.sigmoid(z, True)
                azs.append (a)

            # Do a backpropogate
            gradient_C = (a - y)
            delta = np.multiply (gradient_C, self.sigmoid_prime(zs[-1], True))
            del_biases[-1] += delta
            del_weights[-1] += np.matmul (delta, azs[-2].transpose())

            for l in range (2, self.L + 1):
                sp = self.sigmoid_prime(zs[-l], False)
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
    def sigmoid (self, z, lastLayer):
        if (self.useRELU and not lastLayer):
            return np.maximum(0, z)
        else:
            return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime (self, z, lastLayer):
        if (self.useRELU and not lastLayer):
            return (z > 0).astype(int)
        else:
            sig = self.sigmoid (z, lastLayer)
            sp = np.multiply(sig, 1-sig)
            return sp
