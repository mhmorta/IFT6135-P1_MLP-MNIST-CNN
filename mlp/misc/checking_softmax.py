
import numpy as np


def softmax2(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

x_normal = np.array([-1288.29408504, 710.74971033,  -791.86868885,   543.57083793,
   772.46671785, -1182.81994938, -1660.28336234,   856.3821995,
  1858.68565115,  2229.29081053])
#working
x_working = np.array([0.05649984,  0.01500392, -0.00542593, -0.0224822,  -0.02728796,  0.03694324,
 -0.02031979,  0.00319718,  0.04497563,  0.07353401])

x = x_working
print(softmax2(x))
from sklearn.utils.extmath import softmax
print(softmax([x]))

print([[np.random.normal(0, 1) for _ in range(10)] for _ in range(1)])


#lr, neurons, batch_size=2,4,8,16,32,64,128


w = np.array([['w_11', 'w_12', 'w_13'], ['w_21', 'w_22', 'w_23']])
print('w',w)
print('w^T', w.T)