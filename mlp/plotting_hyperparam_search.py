from matplotlib import pyplot as plt
import numpy as np
import os


reports = [f.replace('.csv', '') for f in os.listdir("reports/hyperparams")]
reports = sorted(reports)
best_model = None
y_pos = np.arange(len(reports))
template = "{0:50}|{1:10}"
print(template.format('Hyperparameters', 'Accuracy'))
print(''.join(['-' for _ in range(60)]))
for idx, hp in enumerate(reports):
    with open("reports/hyperparams/{}.csv".format(hp), "r") as f:
        stats = np.loadtxt(f, delimiter=",", skiprows=1)
        accuracy = (1 - stats[-1, 3]) * 100
        print(template.format(hp, "%.2f" % accuracy))
        if best_model is None or best_model[1] < accuracy:
            best_model = (hp, accuracy)
        plt.bar(idx, accuracy, label=hp)

print('\nBest model with params\n', best_model[0], '\nhas accuracy of', best_model[1])
plt.xticks(y_pos, [r.replace(',m','\nm') for r in reports], rotation=90)
plt.title('Accuracy of different models on validation set')
plt.xlabel('Hypeparameters')
plt.ylabel('Accuracy (%)')
plt.show()