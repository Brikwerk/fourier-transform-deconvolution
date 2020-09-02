import os
import argparse

from skimage import io
from skimage.transform import rescale
from skimage.restoration import estimate_sigma
from skimage.exposure import rescale_intensity, equalize_adapthist, equalize_hist
from skimage import img_as_ubyte, img_as_float64
import numpy as np
import scipy
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('disk_image_path',
                    help="The path to the disk image used to deconvolve the detector image.")
parser.add_argument('detector_image_path',
                    help="The path to the detector image to be deconvolved.")
parser.add_argument('-m', '--magnification', required=False, default=1.0, type=float,
                    help="""How much the disk should be magnificed. Defaults to 1.0.""")
parser.add_argument('-dw', '--detector_width', required=False, default=1770, type=int,
                    help="""The width of a raw, binary detector image, if a binary detector
                    image is specified.""")
parser.add_argument('-dh', '--detector_height', required=False, default=2370, type=int,
                    help="""The height of a raw, binary detector image, if a binary detector
                    image is specified.""")
parser.add_argument('-c', '--correct_distortion', required=False, default=False, type=float,
                    help="""Specifies if distortion correction should be applied. A good
                    starting number for this is -1.5e-5.""")
parser.add_argument('-x', '--shift_x', required=False, default=0, type=int,
                    help="""How much to shift the detector image left or right. Shifting the image
                    will delete the portion of the image shifted. The remainder will be filled
                    with black.""")
parser.add_argument('-y', '--shift_y', required=False, default=0, type=int,
                    help="""How much to shift the detector image up or down. Shifting the image
                    will delete the portion of the image shifted. The remainder will be filled
                    with black.""")
args = parser.parse_args()


def reconstruct(disk, detector, disk_magnification=False):

    if disk_magnification:
        disk = rescale(disk, disk_magnification, anti_aliasing=False, order=0)

    disk_height, disk_width = disk.shape
    detector_height, detector_width = detector.shape

    # Zero padding disk and detector.
    # Padding with the bigger image's size.
    if detector_width + detector_height > disk_height + disk_width:
        pad_max = detector_width + detector_height
    else:
        pad_max = disk_width + disk_height

    # Performing deconvolution
    disk_f = np.fft.fftn(disk, s=[pad_max, pad_max], axes=(0,1))
    detector_f = np.fft.fftn(detector, s=[pad_max, pad_max], axes=(0,1))
    source_f = detector_f / disk_f

    # Converting reconstruction back from fourier domain
    # and saving as an image.
    source = np.fft.ifft2(source_f)
    source = np.fft.fftshift(source)
    source = np.real(source)

    return source


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
    # NOTE: This equation won't be an exact fit for most CR detectors,
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


def shift_image(image, x, y):

    shifted = image

    # X Shifting
    if x < 0: # Shifting image left
        shifted = np.pad(shifted, [(0, 0), (0, np.abs(x))])
        shifted = shifted[:, np.abs(x):]
    else: # Shifting image right
        shifted = np.pad(shifted, [(0, 0), (np.abs(x), 0)])
        shifted = shifted[:, :shifted.shape[1] - np.abs(x)]

    # Y Shifting
    if y > 0: # Shifting image down
        shifted = np.pad(shifted, [(0, np.abs(y)), (0, 0)])
        shifted = shifted[np.abs(y):, :]
    else: # Shifting image up
        shifted = np.pad(shifted, [(np.abs(y), 0), (0, 0)])
        shifted = shifted[:shifted.shape[1] - np.abs(y), :]

    return shifted


def correct_distortion(image, distortion_coefficient):

    height, width = image.shape

    # Distortion coefficients go here
    dist_coeffs = np.zeros((4, 1), np.float64)
    dist_coeffs[0, 0] = distortion_coefficient
    dist_coeffs[1, 0] = 0.0
    dist_coeffs[2, 0] = 0.0
    dist_coeffs[3, 0] = 0.0

    # Transform points, assume unit matrix for camera
    points = np.eye(3, dtype=np.float32)
    points[0, 2] = width/2.0    # Centre X
    points[1, 2] = height/2.0   # Centre Y
    points[0, 0] = 30.          # Focal length X
    points[1, 1] = 30.          # Focal length Y

    undistorted = cv2.undistort(image, points, dist_coeffs)

    return undistorted


if __name__ == "__main__":

    # Loading disk image
    DISK_IMAGE_PATH = args.disk_image_path
    disk = io.imread(DISK_IMAGE_PATH, as_gray=True)
    disk_height, disk_width = disk.shape

    file_name, file_extension = os.path.splitext(args.detector_image_path)
    if file_extension == ".std":
        # Loading raw detector data
        DETECTOR_IMAGE_PATH = args.detector_image_path
        # Detector shift parameters. X -/+ is left/right
        # Y -/+ is down/up
        DETECTOR_X_SHIFT = args.shift_x
        DETECTOR_Y_SHIFT = args.shift_y
        detector_width = args.detector_width
        detector_height = args.detector_height
        detector = np.fromfile(DETECTOR_IMAGE_PATH, dtype="uint16")
        detector = np.flipud(detector.reshape(detector_width, detector_height))
        detector = map_cr_values(detector, kvp=70)
        detector = np.pad(detector, [(0, 0), (400, 0)])
        detector = shift_image(detector, DETECTOR_X_SHIFT, DETECTOR_Y_SHIFT)
    else:
        detector = io.imread(args.detector_image_path, as_gray=True)
        detector = np.pad(detector, [(0, 0), (400, 0)])
        detector = shift_image(detector, DETECTOR_X_SHIFT, DETECTOR_Y_SHIFT)

    if args.correct_distortion:
        detector = correct_distortion(detector, args.correct_distortion)

    if not os.path.isdir("./results"):
        os.mkdir("./results")

    source = reconstruct(disk, detector, disk_magnification=args.magnification)

    source = np.real(source)
    source += np.abs(np.min(source))
    source /= np.average(source)
    source /= np.max(source)
    source = img_as_ubyte(source)

    io.imsave("./results/reconstruction.png", source, check_contrast=False)
