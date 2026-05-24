import numpy as np

def convolution(image_array, filter_array):
    image_array_height, image_array_width = image_array.shape
    filter_array_height, filter_array_width = filter_array.shape
    output_array_height = image_array_height - filter_array_height + 1
    output_array_width = image_array_width - filter_array_width + 1

    output_array = np.zeros((output_array_height, output_array_width))

    