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