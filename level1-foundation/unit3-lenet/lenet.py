import numpy as np


def relu(x):
    return np.maximum(0, x)

def convolution(image_array, filter_array):
    if len(image_array.shape) == 3:
        height = image_array.shape[1]
        width = image_array.shape[2]
        output_height = height - filter_array.shape[0] + 1
        output_width = width - filter_array.shape[1] + 1
        output = np.zeros((output_height, output_width))
        for channel in range(image_array.shape[0]):
            output += convolution(image_array[channel], filter_array)
        return output

    image_array_height, image_array_width = image_array.shape
    filter_array_height, filter_array_width = filter_array.shape
    output_array_height = image_array_height - filter_array_height + 1
    output_array_width = image_array_width - filter_array_width + 1

    output_array = np.zeros((output_array_height, output_array_width))

    for i in range(output_array_height):
        for j in range(output_array_width):
            patch = image_array[i:i+filter_array_height, j:j+filter_array_width]
            output_array[i][j] = np.sum(patch * filter_array)
    
    return output_array


def pooling(feature_map, pool_size):
    feature_map_height, feature_map_width = feature_map.shape
    output_height = feature_map_height // pool_size
    output_width = feature_map_width // pool_size
    pool_map = np.zeros((output_height,output_width))

    for i in range(output_height):
        for j in range(output_width):
            block = feature_map[i* pool_size:i*pool_size + pool_size,j* pool_size:j*pool_size+pool_size]
            pool_map[i][j] = np.max(block)

    return pool_map

def conv_layer(input_data, filters):
    results = []
    num_filters = filters.shape[0]
    for i in range(num_filters):
        convolution_result = convolution(input_data, filters[i])
        result = relu(convolution_result)
        results.append(result)
    return np.array(results)

def pool_layer(feature_maps, pool_size):
    results = []
    for feature_map in feature_maps:
        pooling_result = pooling(feature_map, pool_size)
        results.append(pooling_result)
    return np.array(results)

def features_to_network(pool_maps, weights, bias):
    flat = pool_maps.flatten()
    output = np.dot(flat, weights) + bias
    result = relu(output)
    return result 

def forward_pass(image, filters1, filters2, weights, bias):
    first_pass  = conv_layer(image, filters1)
    first_pass_pool = pool_layer(first_pass, 2)
    second_pass = conv_layer(first_pass_pool, filters2)
    second_pass_pool = pool_layer(second_pass,2)
    result = features_to_network(second_pass_pool, weights, bias)
    return result

def one_hot(label):
    result = np.zeros(10)
    result[label] = 1
    return result
    
def loss(output, correct):
    return np.mean((output - correct)**2)

def loss_gradient(output, correct):
    # measures how much each output score contributed to the loss
    # derivative of mean squared error loss: d(loss)/d(output)
    # loss = mean((output - correct)^2) = (1/n) * sum((output - correct)^2)
    # derivative = 2 * (output - correct) / n
    # where n = len(output) = 10 (one per digit)
    return 2* (output - correct)/len(output)

def fc_backprop(flat, weights, loss_grad):
    # FC layer forward was: output = np.dot(flat, weights) + bias
    # we need 3 gradients using chain rule: d(loss)/d(x) = d(loss)/d(output) × d(output)/d(x)

    # WEIGHTS GRADIENT
    # simple case: y = x·w → dy/dw = x
    # so: d(output)/d(weights) = flat
    # by the chain rule: d(loss)/d(weights) = d(loss)/d(output) × d(output)/d(weights)
    #                                       = loss_grad × flat
    # shapes: (400,1) · (1,10) = (400,10) — same shape as weights!
    weights_gradient = np.dot(flat.reshape(-1,1), loss_grad.reshape(1,-1))

    # BIAS GRADIENT
    # simple case: y = x + b → dy/db = 1
    # so: d(output)/d(bias) = 1
    # by the chain rule: d(loss)/d(bias) = d(loss)/d(output) × d(output)/d(bias)
    #                                    = loss_grad × 1 = loss_grad
    # shape: (10,) — same shape as bias!
    bias_gradient = loss_grad

    # INPUT GRADIENT
    # simple case: y = x·w → dy/dx = w
    # so: d(output)/d(flat) = weights
    # by the chain rule: d(loss)/d(flat) = d(loss)/d(output) × d(output)/d(flat)
    #                                    = loss_grad × weights.T
    # shapes: (10,) · (10,400) = (400,) — same shape as flat!
    input_gradient = np.dot(loss_grad, weights.T)

    return weights_gradient, bias_gradient, input_gradient