from matplotlib import pyplot as plt
import numpy as np
import os

reports = [f.replace('.csv', '') for f in os.listdir("reports/hyperparams")]

best_model = None
y_pos = np.arange(len(reports))
for idx, hp in enumerate(reports):
    with open("reports/hyperparams/{}.csv".format(hp), "r") as f:
        stats = np.loadtxt(f, delimiter=",", skiprows=1)
        accuracy = (1 - stats[-1, 3]) * 100
        print(hp, '\n', accuracy)
        if best_model is None or best_model[1] < accuracy:
            best_model = (hp, accuracy)
        plt.bar(idx, accuracy, label=hp)

print('Best model with params ', best_model[0], ' has accuracy of ', best_model[1])
plt.xticks(y_pos, [r.replace(',m','\nm') for r in reports], rotation=90)
#plt.legend()
plt.title('Accuracy of different models on validation set')
plt.xlabel('Hypeparameters')
plt.ylabel('Accuracy (%)')
plt.show()