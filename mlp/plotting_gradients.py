from matplotlib import pyplot as plt
import numpy as np


with open("reports/gradients/{}".format('gradient_epochs=10,hidden_dims=[700,500],mu=0.3112553202446868,batch_size=1,validate_gradient=True,weight_init=glorot.csv'), "r") as f:
    stats = np.loadtxt(f, delimiter=",", skiprows=1)
    plt.plot(stats[:, 2], 'o')
plt.xticks(range(len(stats[:,0])), stats[:, 0], rotation='vertical')
plt.title('Maximum difference between gradients')
plt.xlabel('N')
plt.ylabel('Difference')
plt.show()