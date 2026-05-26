import numpy as np
from lenet import convolution, pooling, conv_layer, pool_layer, relu, features_to_network, forward_pass
from tensorflow.keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


five = train_images[0] / 255.0  # first image
print(train_labels[0])   # what digit is it?
print(five.shape)        # should be (28, 28)



# Random filters for conv layer 1 — 6 filters, each 3x3
filters1 = np.random.randn(6, 3, 3)
# Random filters for conv layer 2 — 16 filters, each 3x3
filters2 = np.random.randn(16, 3, 3)


weights = np.random.randn(400, 10)
bias = np.random.randn(10)


output = forward_pass(five, filters1, filters2, weights, bias)
print(output)