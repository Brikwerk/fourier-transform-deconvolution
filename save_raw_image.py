import os
import argparse

import numpy as np
from skimage import img_as_ubyte
from skimage import io


def map_cr_values(binary_image, kvp=70):
    """Maps values from a binary CR image to a float image
    based off of a calculation involving the kVp used to
    generate the image.
    
    :param binary_image: A Numpy array containing the CR image values
    :type binary_image: numpy.ndarray
    :param kvp: The kVp used to generate the CR binary image, defaults to 70
    :type kvp: int, optional
    :return: A float image with values ranging from (-1, 1)
    :rtype: numpy.ndarray
    """

    # Getting C value for mapping equation
    # NOTE: This equation won't be an exact fit for most CR images,
    # however, it should be good enough for the purposes of
    # calibration within this library.
    C = (-0.0739 * np.power(kvp, 2)) + (15.408 * kvp) + 301.17
    # Applying CR mapping equation
    map_values = np.subtract(binary_image, C)
    map_values = np.divide(map_values, 1024)
    map_values = np.power(10, map_values)
    # Normalizing values to float image range (-1, 1)
    map_values = np.divide(map_values, np.max(map_values))

    return map_values


def raw_to_image(file_location, height, width):

    image = np.fromfile(file_location, dtype="uint16")
    image = np.flipud(image.reshape(width, height))
    image = map_cr_values(image, kvp=70)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file_location', type=str,
                        help='The location of the raw file to load')
    parser.add_argument('width', type=int,
                        help='The width of the raw file')
    parser.add_argument('height', type=int,
                        help='The height of the raw file')

    args = parser.parse_args()
    image = raw_to_image(args.file_location, args.height, args.width)
    image = img_as_ubyte(image)

    io.imsave(args.file_location + ".png", image)
