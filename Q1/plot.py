import numpy as np
import matplotlib.pyplot as plt


def make_acc_curve (X, Y1, Y2, Y3, fileName="Q1/plots/sample.png"):
    markers = ['y-', 'b-', 'g-']
    fig = plt.figure(1)
    line1, = plt.plot(X, Y1, markers[0], label="Training")
    line2, = plt.plot(X, Y2, markers[1], label="Validation")
    line3, = plt.plot(X, Y3, markers[2], label="Test")
    
    legend = plt.legend(handles=[line1, line2, line3])

    plt.title("Accuracy vs Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("% Accuracy")

    plt.xlim(left=0)
    # plt.ylim(bottom=0)
    # plt.ylim(top=100)

    fig.savefig(fileName)
    plt.show()