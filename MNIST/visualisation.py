from matplotlib import pyplot as plt
import numpy as np

reports = [
    ('cnn', 'cnn_report.csv'),
    ('mlp', 'report_epochs=10,hidden_dims=[500,600],mu=0.1,batch_size=32,weight_init=glorot.csv')
]

for report in reports:
    weight_init, file_name = report
    with open("reports/{}".format(file_name), "r") as f:
        stats = np.loadtxt(f, delimiter=",", skiprows=1)
        epochs = stats[:, 0]
        train = stats[:, 1]
        plt.plot(epochs, train, label=weight_init)

plt.legend()
plt.grid()
plt.title('Train error, CNN vs MLP')
plt.xlabel('Epoch')
plt.ylabel('Train error')
plt.show()