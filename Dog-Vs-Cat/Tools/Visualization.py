import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

def plots(nll_train, nll_valid, acc_train, acc_valid ):
    plt.figure(1)
    plt.plot(range(len(nll_train)), nll_train, label="Loss train" )
    plt.plot(range(len(nll_valid)), nll_valid,   label="Loss validation" )
    plt.figure(2)
    plt.plot(range(len(acc_train)), acc_train, label="Accuracy train" )
    plt.plot(range(len(acc_valid)), acc_valid,   label="Accuracy validation" )
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes=('Cat', 'Dog'),
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()