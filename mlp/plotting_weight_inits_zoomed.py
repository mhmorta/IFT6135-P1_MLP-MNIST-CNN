from matplotlib import pyplot as plt
import numpy as np

reports = [('normal', 'report_epochs=10,hidden_dims=[500,600],mu=0.01,batch_size=32,weight_init=normal.csv'),
           ('glorot', 'report_epochs=10,hidden_dims=[500,600],mu=0.01,batch_size=32,weight_init=glorot.csv')
]

for report in reports:
    weight_init, file_name = report
    start_epoch_idx = 3
    with open("reports/weight_inits/{}".format(file_name), "r") as f:
        stats = np.loadtxt(f, delimiter=",", skiprows=1)
        train = stats[start_epoch_idx:, 2]
        print(weight_init, '\n', train)
        plt.plot(train, label=weight_init)
plt.xticks(range(len(train)), range(start_epoch_idx + 1, len(stats) + 1))
plt.legend()
plt.title('Average loss zoomed from epoch {} (on training data, mu=0.01)'.format(start_epoch_idx+1))
plt.xlabel('Epoch')
plt.ylabel('Average loss')
plt.grid()
plt.show()