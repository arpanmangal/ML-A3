import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

class_names = [str(x) for x in range(10)]
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Plotting Code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def make_confusion_matrix (trueY, predictions, fileName="Q2/plots/confusion.png", show=True):
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(trueY, predictions, classes=class_names,
                        title='Confusion Matrix')

    plt.savefig(fileName, dpi=100)
    plt.show()


def make_line_curve (Y, X=None, Xlabel="X", Ylabel="Y", marker="g-", fileName="Q2/plots/sample.png", title="Line Plot", miny=None, maxy=None):
    if X is None:
        X = [int(x) for x in range(len(Y))]
    fig = plt.figure(1)
    plt.plot(X, Y, marker)
    plt.title(title)
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.xlim(left=0)
    if miny is not None:
        plt.ylim(bottom=miny)
    if maxy is not None:
        plt.ylim(top=maxy)
    fig.savefig(fileName)
    plt.show()