"""
The class of Neural Network
"""
import numpy as np

class NNetwork:
    def __init__ (self, num_input, sizes, num_output, batch_size):
        self.sizes = sizes
        self.sizes.append (num_output)
        self.sizes.insert(0, num_input)
        self.batch_size = batch_size

        self.L = len(sizes)

        # Make the Weight and Biases vectors for each layer
        self.weights = [None] * self.L
        self.biases = [None] * self.L
        for l in range(1, self.L):
            self.weights[l] = np.random.random((sizes[l], sizes[l-1]))
            self.biases[l] = np.random.random((sizes[l],))
            
        for b in self.biases:
            if b is not None:
                print (b.shape)
        for w in self.weights:
            if w is not None:
                print (w.shape)


    def predict (self, X):
        return np.array(list(map(self.feedforward, X)))

    def feedforward (self, input):
        # Run the neural network on the input and reuturn the output
        a = [None] * self.L
        a[0] = input
        for l in range(1, self.L):
            z = np.matmul(self.weights[l], a[l-1]) + self.biases[l]
            a[l] = self.sigmoid(z)

        return np.argmax(a[-1])


    
    # Other Functions
    def accuracy (self, trueY, predictions):
        return np.sum(trueY == predictions) / len(trueY)

    # Helper Functions
    def sigmoid (self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime (self, z):
        sig = self.sigmoid (z)
        return np.multiply(sig, 1-sig)
