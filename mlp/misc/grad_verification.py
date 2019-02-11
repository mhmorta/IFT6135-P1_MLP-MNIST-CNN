import numpy as np

#b2 gradient diff:  1.8630384615157297e-11
#------------- error ---------------
#b2 gradient error:  0.07087044882388992
#b2 estimated gradient:  -1.2212453274237143e-10
#b2 calculated gradient:  -1.4075491735752873e-10

calculated_gradient = -1.4075491735752873e-10
estimated_gradient = -1.2212453274237143e-10
diff = np.abs(calculated_gradient - estimated_gradient)
summ = np.abs(calculated_gradient + estimated_gradient)
grad_threshold = 0.02

grad_error = diff / summ if summ != 0 else 0
if grad_error > grad_threshold:
    print('Error')