import numpy as np

def convolution(image_array, filter_array):
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