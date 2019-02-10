import matplotlib.pyplot as plt


def plots(nll_train, nll_valid, acc_train, acc_valid ):
    plt.figure(1)
    plt.plot(range(len(nll_train)), nll_train, label="Loss train" )
    plt.plot(range(len(nll_valid)), nll_valid,   label="Loss validation" )
    plt.figure(2)
    plt.plot(range(len(acc_train)), acc_train, label="Accuracy train" )
    plt.plot(range(len(acc_valid)), acc_valid,   label="Accuracy validation" )
    plt.legend()
    plt.show()