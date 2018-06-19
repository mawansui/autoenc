import numpy as np
from math import ceil

real_y_string = "0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
real_y_string_array = real_y_string.split()
real_y = [int(x) for x in real_y_string_array]
predicted_y_string_0 = """1.23971179e-02
1.01962769e+00
-6.26208633e-03
9.65471625e-01
-5.77314943e-03
6.27782196e-04
1.03154159e+00
1.23146139e-02
-2.00711358e-02
5.39081022e-02
2.06600707e-02
-6.41515106e-03
1.41123114e-02
-5.72123006e-02
-2.71934830e-02
-9.77768190e-03
5.81393540e-02
9.64527786e-01
-5.02191251e-04
1.08519699e-02
8.05304386e-03
1.44751156e-02
4.98285964e-02
-7.10927788e-03
9.30608064e-03
-2.16164403e-02
-7.95101281e-03
-1.23267490e-02
8.51051882e-05
6.95798248e-02
2.04890966e-06
2.57675368e-02
-1.49920583e-02
-9.19246580e-03
-1.79707669e-02
-1.97932571e-02
8.44411273e-03
2.29292046e-02
4.89210710e-03
5.36463596e-03
1.87242143e-02
-1.71295498e-02"""

predicted_y_string = """-4.08922508e-03
9.93695259e-01
8.61454755e-03
9.83865082e-01
6.65684044e-03
9.86872613e-03
1.00501549e+00
-1.42547488e-03
-7.23741949e-03
-6.73836190e-03
8.50594137e-04
-2.46273982e-03
-8.13164096e-03
9.07537900e-03
7.75153190e-03
1.26303975e-02
1.30188596e-02
9.77150321e-01
-1.86530733e-03
-7.90728908e-03
2.74740160e-05
9.66843404e-03
-3.73860449e-03
-3.66764702e-03
-7.46763777e-03
-4.64862771e-03
-7.59895518e-03
1.09738531e-03
-5.49264252e-04
-2.56459042e-03
-1.10478066e-02
1.13283368e-02
9.01481695e-03
-4.46764566e-03
-6.09925669e-03
4.71433997e-03
-1.30766071e-03
5.62896207e-03
-6.33985689e-03
5.31008467e-04
8.35023075e-03
-7.92803243e-03"""

predicted_y_string_array = predicted_y_string.split("\n")
predicted_y = [float(y) for y in predicted_y_string_array]


final_predicted_y = []
for item in predicted_y:
	if item > 0.5:
		final_predicted_y.append(1)
	else:
		final_predicted_y.append(0)

print("The results:")
print("Real Y values: {}".format(real_y))
print("Pred.Y values: {}".format(final_predicted_y))
# print(predicted_y)

print('\n\n\n')

for real, predicted in zip(real_y, predicted_y):
	print("Real: {}\nPred: {}".format(real, predicted))