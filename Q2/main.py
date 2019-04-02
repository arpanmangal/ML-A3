"""
Main Controller
"""


import sys
import time
from read import *
from nnetwork import NNetwork
from plot import make_confusion_matrix, make_line_curve

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def evaluate_model (NNet, data, print_accuracy=True, conf_matrix=True, savePath="Q2/plots/confusionMatrix.png"):
    # Evaluate the learned model on the data
    # Plot the confusion matrix
    Y = NNet.getTrueY(data)
    predictions = NNet.predict (data)
    accuracy = NNet.accuracy (Y, predictions) * 100
    if print_accuracy:
        print ("Final Accuracy: %.2f %%" % accuracy)
    if conf_matrix:
        make_confusion_matrix (Y, predictions, fileName=savePath, show=conf_matrix)

    return accuracy
        

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print ("Go Away")
        exit(1)

    if (sys.argv[1] == 'a'):
        # Convert to the one-hot encoding
        gen_one_hot_data ('data/poker/poker-hand-training.data', 'data/poker/train.data')
        gen_one_hot_data ('data/poker/poker-hand-testing.data', 'data/poker/test.data')

    elif (sys.argv[1] == 'b'):
        # Read the data
        data = read_one_hot_data ('data/poker/train.data')

        # Train the neural network
        start_time = time.time()
        NNet = NNetwork (85, [20], 10, 1)
        accuracies, losses = NNet.train(data, 0.1)
        print ("Training time: %.2f secs" % (time.time() - start_time))

        make_line_curve (accuracies, Xlabel="Epochs", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/accuracy.png", title="Accuracy vs. #Epochs", miny=0, maxy=100)
        make_line_curve (losses, Xlabel="Epochs", Ylabel="Loss", marker='m-', fileName="Q2/plots/loss.png", title="Loss vs. #Epochs", miny=0)

        evaluate_model (NNet, data, savePath="Q2/plots/confusionMatrix.png")

    elif (sys.argv[1] == 'c'):
        # Experiments with single hidden layers
        eta = 0.1
        hidden_units = [5, 10, 15, 20, 25]

        # Read the data
        trainData = read_one_hot_data ('data/poker/train.data')
        # testData = read_one_hot_data ('data/poker/test.data')

        # The metrics
        testAccuracies = []
        trainAccuracies = []
        trainingTimes = []
        for size in hidden_units:
            # Train the neural network
            start_time = time.time()
            NNet = NNetwork (85, [size], 10, 1)
            accuracies, losses = NNet.train(trainData, 0.1, silent=True)
            trainingTimes.append(time.time() - start_time)
            print ("Size %d done | Time Take: %.2f secs" % (size, time.time() -start_time))

            save_conf_matrix = "Q2/plots/confMatrix-c-" + str(size)
            trainAcc = evaluate_model (NNet, trainData, conf_matrix=False, savePath=save_conf_matrix)
            trainAccuracies.append(trainAcc)

        print (trainingTimes)
        print (trainAccuracies)

        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartCTrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainingTimes, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Time (secs)", marker='g-', fileName="Q2/plots/PartCTrainTime.png", title="Training time vs. Size of Hidden Layer")

    elif (sys.argv[1] == 'd'):
        # Experiments with two hidden layers
        eta = 0.1
        hidden_units = [[x,x] for x in [5, 10, 15, 20, 25]]

        # Read the data
        trainData = read_one_hot_data ('data/poker/train.data')
        # testData = read_one_hot_data ('data/poker/test.data')

        # The metrics
        testAccuracies = []
        trainAccuracies = []
        trainingTimes = []
        for size in hidden_units:
            # Train the neural network
            start_time = time.time()
            NNet = NNetwork (85, size, 10, 1)
            accuracies, losses = NNet.train(trainData, 0.1, silent=True, max_epochs=5)
            trainingTimes.append(time.time() - start_time)
            print ("Size %d done | Time Take: %.2f secs" % (size[0], time.time() -start_time))

            save_conf_matrix = "Q2/plots/confMatrix-d-" + str(size[0])
            trainAcc = evaluate_model (NNet, trainData, conf_matrix=False, savePath=save_conf_matrix)
            trainAccuracies.append(trainAcc)

        print (trainingTimes)
        print (trainAccuracies)

        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartDTrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainingTimes, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Time (secs)", marker='g-', fileName="Q2/plots/PartDTrainTime.png", title="Training time vs. Size of Hidden Layer")

    elif (sys.argv[1] == 'e'):
        eta = 0.1
        hidden_units = [[x,x] for x in [10, 20]]

        # Read the data
        trainData = read_one_hot_data ('data/poker/train.data')
        # testData = read_one_hot_data ('data/poker/test.data')

        # The metrics
        testAccuracies = []
        trainAccuracies = []
        trainingTimes = []
        for size in hidden_units:
            print (size, size[0])
            # continue
            # Train the neural network
            start_time = time.time()
            NNet = NNetwork (85, size, 10, 1)
            accuracies, losses = NNet.train(trainData, 0.1, silent=False, max_epochs=5, adaptive_eta=True)
            trainingTimes.append(time.time() - start_time)
            print ("Size %d done | Time Take: %.2f secs" % (size[0], time.time() -start_time))

            save_conf_matrix = "Q2/plots/confMatrix-e-" + str(size[0])
            trainAcc = evaluate_model (NNet, trainData, conf_matrix=False, savePath=save_conf_matrix)
            trainAccuracies.append(trainAcc)

        print (trainingTimes)
        print (trainAccuracies)

        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartETrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainingTimes, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Time (secs)", marker='g-', fileName="Q2/plots/PartETrainTime.png", title="Training time vs. Size of Hidden Layer")
    
    
    elif (sys.argv[1] == 'f'):
        eta = 0.1
        hidden_units = [[x,x] for x in [10, 20]]

        # Read the data
        trainData = read_one_hot_data ('data/poker/train.data')
        # testData = read_one_hot_data ('data/poker/test.data')

        # The metrics
        testAccuracies = []
        trainAccuracies = []
        trainingTimes = []
        for size in hidden_units:
            print (size, size[0])
            # continue
            # Train the neural network
            start_time = time.time()
            NNet = NNetwork (85, size, 10, 1, useRELU=True)
            accuracies, losses = NNet.train(trainData, 0.1, silent=False, max_epochs=5, adaptive_eta=True)
            trainingTimes.append(time.time() - start_time)
            print ("Size %d done | Time Take: %.2f secs" % (size[0], time.time() -start_time))

            save_conf_matrix = "Q2/plots/confMatrix-e-" + str(size[0])
            trainAcc = evaluate_model (NNet, trainData, conf_matrix=False, savePath=save_conf_matrix)
            trainAccuracies.append(trainAcc)

        print (trainingTimes)
        print (trainAccuracies)

        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartETrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainingTimes, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Time (secs)", marker='g-', fileName="Q2/plots/PartETrainTime.png", title="Training time vs. Size of Hidden Layer")

    elif (sys.argv[1] == 't'):
        # make_line_curve ([10,2,-4,4,5,2], marker='m-', miny=2, maxy=4)
        
        hidden_units = [5, 10, 15, 20, 25]
        trainAccuracies = [43.00279888044783, 46.0375849660136, 49.95201919232307, 49.8360655737705, 47.76489404238305]
        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartCTrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)

    else:
        print ("Go Away")
        exit(1)