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
    print (sys.argv)
    if (len(sys.argv) < 2):
        print ("Go Away")
        exit(1)

    if (sys.argv[1] == 'p'):
        # Convert to the one-hot encoding
        gen_one_hot_data (sys.argv[2], sys.argv[3]) #  'data/poker/poker-hand-training.data', 'data/poker/train.data')
        gen_one_hot_data (sys.argv[4], sys.argv[5]) # 'data/poker/poker-hand-testing.data', 'data/poker/test.data')

    elif (sys.argv[1] == 'b'):
        # Read the data
        data = read_one_hot_data ('data/poker/train.data')
        testData = read_one_hot_data ('data/poker/test.data')

        # Train the neural network
        start_time = time.time()
        NNet = NNetwork (85, [20], 10, 1)
        accuracies, losses = NNet.train(data, 0.1)
        print ("Training time: %.2f secs" % (time.time() - start_time))

        make_line_curve (accuracies, Xlabel="Epochs", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartB/accuracy.png", title="Accuracy vs. #Epochs", miny=0, maxy=100)
        make_line_curve (losses[1:], Xlabel="Epochs", Ylabel="Loss", marker='m-', fileName="Q2/plots/PartB/loss.png", title="Loss vs. #Epochs", miny=0)

        evaluate_model (NNet, testData, savePath="Q2/plots/PartB/confusionMatrixTest.png")
        evaluate_model (NNet, data, savePath="Q2/plots/PartB/confusionMatrixTrain.png")

    elif (sys.argv[1] == 'c'):
        # Experiments with single hidden layers
        eta = 0.1
        hidden_units = [5, 10, 15, 20, 25]

        # Read the data
        trainData = read_one_hot_data ('data/poker/train.data')
        testData = read_one_hot_data ('data/poker/test.data')

        # The metrics
        testAccuracies = []
        trainAccuracies = []
        trainingTimes = []
        for size in hidden_units:
            # Train the neural network
            start_time = time.time()
            NNet = NNetwork (85, [size], 10, 1)
            accuracies, losses = NNet.train(trainData, 0.1, silent=False)
            trainingTimes.append(time.time() - start_time)
            print ("Size %d done | Time Take: %.2f secs" % (size, time.time() -start_time))

            save_conf_matrix = "Q2/plots/PartC/confMatrix-c-" + str(size)
            trainAcc = evaluate_model (NNet, trainData, conf_matrix=False, savePath=save_conf_matrix)
            testAcc = evaluate_model (NNet, testData, savePath=save_conf_matrix)
            trainAccuracies.append(trainAcc)
            testAccuracies.append(testAcc)

        print (trainingTimes)
        print (trainAccuracies)
        print (testAccuracies)

        make_line_curve (testAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartC/PartCTestAcc.png", title="Test Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartC/PartCTrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainingTimes, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Time (secs)", marker='g-', fileName="Q2/plots/PartC/PartCTrainTime.png", title="Training time vs. Size of Hidden Layer")

    elif (sys.argv[1] == 'd'):
        # Experiments with two hidden layers
        eta = 0.1
        hidden_units = [[x,x] for x in [5, 10, 15, 20, 25]]

        # Read the data
        trainData = read_one_hot_data ('data/poker/train.data')
        testData = read_one_hot_data ('data/poker/test.data')

        # The metrics
        testAccuracies = []
        trainAccuracies = []
        trainingTimes = []
        for size in hidden_units:
            # Train the neural network
            start_time = time.time()
            NNet = NNetwork (85, size, 10, 1)
            accuracies, losses = NNet.train(trainData, 0.1, silent=True)
            trainingTimes.append(time.time() - start_time)
            print ("Size %d done | Time Take: %.2f secs" % (size[0], time.time() -start_time))

            save_conf_matrix = "Q2/plots/PartD/confMatrix-d-" + str(size[0])
            trainAcc = evaluate_model (NNet, trainData, conf_matrix=False, savePath=save_conf_matrix)
            testAcc = evaluate_model (NNet, testData, savePath=save_conf_matrix)
            trainAccuracies.append(trainAcc)
            testAccuracies.append(testAcc)

        print (trainingTimes)
        print (trainAccuracies)
        print (testAccuracies)

        make_line_curve (testAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartD/PartDTestAcc.png", title="Test Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartD/PartDTrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainingTimes, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Time (secs)", marker='g-', fileName="Q2/plots/PartD/PartDTrainTime.png", title="Training time vs. Size of Hidden Layer")

    elif (sys.argv[1] == 'e'):
        eta = 0.1
        hidden_units = [[x,x] for x in [10, 20]]

        # Read the data
        trainData = read_one_hot_data ('data/poker/train.data')
        testData = read_one_hot_data ('data/poker/test.data')

        # The metrics
        testAccuracies = []
        trainAccuracies = []
        trainingTimes = []
        for size in hidden_units:
            # Train the neural network
            start_time = time.time()
            NNet = NNetwork (85, size, 10, 1)
            accuracies, losses = NNet.train(trainData, 0.1, silent=False, adaptive_eta=True)
            trainingTimes.append(time.time() - start_time)
            print ("Size %d done | Time Take: %.2f secs" % (size[0], time.time() -start_time))

            save_conf_matrix = "Q2/plots/PartE/confMatrix-e-" + str(size[0])
            trainAcc = evaluate_model (NNet, trainData, conf_matrix=False, savePath=save_conf_matrix)
            testAcc = evaluate_model (NNet, testData, savePath=save_conf_matrix)
            trainAccuracies.append(trainAcc)
            testAccuracies.append(testAcc)

        print (trainingTimes)
        print (trainAccuracies)
        print (testAccuracies)

        make_line_curve (testAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartE/PartETestAcc.png", title="Test Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartE/PartETrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainingTimes, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Time (secs)", marker='g-', fileName="Q2/plots/PartE/PartETrainTime.png", title="Training time vs. Size of Hidden Layer")
    
    elif (sys.argv[1] == 'f'):
        eta = 0.1
        hidden_units = [[x,x] for x in [10, 20]]

        # Read the data
        trainData = read_one_hot_data ('data/poker/train.data')
        testData = read_one_hot_data ('data/poker/test.data')

        # The metrics
        testAccuracies = []
        trainAccuracies = []
        trainingTimes = []
        for size in hidden_units:
            # Train the neural network
            start_time = time.time()
            NNet = NNetwork (85, size, 10, 1, useRELU=True)
            accuracies, losses = NNet.train(trainData, 0.1, silent=False, adaptive_eta=True)
            trainingTimes.append(time.time() - start_time)
            print ("Size %d done | Time Take: %.2f secs" % (size[0], time.time() -start_time))

            save_conf_matrix = "Q2/plots/PartF/confMatrix-e-" + str(size[0])
            trainAcc = evaluate_model (NNet, trainData, conf_matrix=False, savePath=save_conf_matrix)
            testAcc = evaluate_model (NNet, testData, savePath=save_conf_matrix)
            trainAccuracies.append(trainAcc)
            testAccuracies.append(testAcc)

        print (trainingTimes)
        print (trainAccuracies)
        print (testAccuracies)

        make_line_curve (testAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartF/PartFTestAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartF/PartFTrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)
        make_line_curve (trainingTimes, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Time (secs)", marker='g-', fileName="Q2/plots/PartF/PartFTrainTime.png", title="Training time vs. Size of Hidden Layer")

    elif (sys.argv[1] == 't'):
        # make_line_curve ([10,2,-4,4,5,2], marker='m-', miny=2, maxy=4)
        # For testing the plot function
        hidden_units = [5, 10, 15, 20, 25]
        trainAccuracies = [43.00279888044783, 46.0375849660136, 49.95201919232307, 49.8360655737705, 47.76489404238305]
        make_line_curve (trainAccuracies, X=hidden_units, Xlabel="Hidden Layer Size", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartCTrainAcc.png", title="Training Accuracy vs. Size of Hidden Layer", miny=0, maxy=100)

    elif (sys.argv[1] == 'nn'):
        # Read from the file the nn architecture
        afile = open(sys.argv[2], 'r')
        inSize = int(afile.readline())
        outSize = int(afile.readline())
        batchSize = int(afile.readline())
        H = int(afile.readline())
        sizes = [int(s) for s in  afile.readline().strip('\n').split(' ')]
        useRELU = (afile.readline().strip('\n')) == 'relu'
        adaptive_eta = (afile.readline().strip('\n')) == 'variable'

        print (inSize, outSize, batchSize, H, sizes, useRELU, adaptive_eta)
        # exit(0)
        # Read the data
        data = read_one_hot_data (sys.argv[3])
        testData = read_one_hot_data (sys.argv[4])

        # Train the neural network
        start_time = time.time()
        NNet = NNetwork (inSize, sizes, outSize, batchSize, useRELU=useRELU)
        accuracies, losses = NNet.train(data, 0.1, adaptive_eta=adaptive_eta)
        print ("Training time: %.2f secs" % (time.time() - start_time))

        make_line_curve (accuracies, Xlabel="Epochs", Ylabel="Accuracy (%)", marker='b-', fileName="Q2/plots/PartNN/accuracy.png", title="Accuracy vs. #Epochs", miny=0, maxy=100)
        make_line_curve (losses[1:], Xlabel="Epochs", Ylabel="Loss", marker='m-', fileName="Q2/plots/PartNN/loss.png", title="Loss vs. #Epochs", miny=0)

        evaluate_model (NNet, testData, savePath="Q2/plots/PartNN/confusionMatrixTest.png")
        evaluate_model (NNet, data, savePath="Q2/plots/PartNN/confusionMatrixTrain.png")


    else:
        print ("Go Away")
        exit(1)