import numpy as np
from lenet import convolution, pooling


three = np.array([
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 1, 1, 0]
])

hotizontal_edge_filter = np.array([
    [+1, +1, +1],
    [0, 0, 0],
    [-1, -1, -1]
])

convolution_pass = convolution(three, hotizontal_edge_filter)
pooling_pass = pooling(convolution_pass, 2)

print("After convolution:")
print(convolution_pass)
print("After pooling:")
print(pooling_pass)