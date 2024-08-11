import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Dropout

# Task 1 (2 marks)
import numpy as np


def image_statistics(image, darkness):
     """Return a dictionary with the following statistics about the image. Assume that
     the image is a colour image with three channels.
     - resolution: a tuple of the form (number_rows, number_columns).
     - dark_pixels: a tuple of three elements, one per channel, where each element
          shows the number of channel values lower than the given darkness value.

     Parameters:
     image (numpy.ndarray): The input image as a 3D numpy array.
     darkness (int): The threshold to consider a pixel as dark.

     Returns:
     dict: A dictionary containing the resolution and dark_pixels information.
     """
     # Calculate resolution
     resolution = image.shape[:2]

     # Count dark pixels in each channel
     dark_pixels = tuple((image[:, :, i] < darkness).sum() for i in range(3))

     # Return the result as a dictionary
     return {'resolution': resolution, 'dark_pixels': dark_pixels}


# Task 2 (2 marks)
def bounding_box(image, top_left, bottom_right):
     """Return an extract of the image determined by the bounding box, where the bounding box
     is the (row, column) positions of the pixels at the top left and bottom right of the box.

     Parameters:
     image (numpy.ndarray): The input image as a 3D numpy array.
     top_left (tuple): The (row, column) position of the top-left corner of the bounding box.
     bottom_right (tuple): The (row, column) position of the bottom-right corner of the bounding box.

     Returns:
     numpy.ndarray: The cropped image within the bounding box.
     """
     # Extract the region using slicing
     # Note: bottom_right indices are increased by 1 to include the bottom_right pixel in the slice
     roi = image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
     return roi

# Task 3 (2 marks)
def build_deep_nn(rows, columns, channels, num_hidden, hidden_sizes, dropout_rates, output_size, output_activation):
     model = Sequential()
     model.add(Flatten(input_shape=(rows, columns, channels)))

     for i in range(num_hidden):
          model.add(Dense(hidden_sizes[i], activation='relu'))
          if dropout_rates[i] > 0:  # Assuming dropout_rates is a tuple/list with length=num_hidden
               model.add(Dropout(dropout_rates[i]))

     model.add(Dense(output_size, activation=output_activation))
     return model


if __name__ == "__main__":
     import doctest
     doctest.testmod()