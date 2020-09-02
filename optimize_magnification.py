"""
This script attempts to optimize the magnification such that the best reconstruction
is produced. 

---
How It Works
---

This is achieved by observing an image patch on the deconvolved image
and measuring the width of the histogram. Larger histograms indicate more noise,
which means the magnification value is off. To optimize for this value, this script
iterates over a range of magnification values and saves the deconvolution with the
smallest histogram width.
"""

import os
import argparse

import numpy as np
from skimage import io
from skimage import img_as_ubyte

from reconstruct import map_cr_values
from reconstruct import shift_image
from reconstruct import reconstruct
from reconstruct import correct_distortion


parser = argparse.ArgumentParser()
parser.add_argument('disk_image_path',
                    help="The path to the disk image used to deconvolve the detector image.")
parser.add_argument('detector_image_path',
                    help="The path to the detector image to be deconvolved.")
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

parser.add_argument('-i', '--iterations', required=False, default=20, type=int,
                    help="""How many optimizations iterations should be run. Defaults to 20.""")
parser.add_argument('-s', '--starting_magnification', required=False, default=0.1, type=float,
                    help="""The magnification value to start optimization at.""")
parser.add_argument('-e', '--ending_magnification', required=False, default=2.0, type=float,
                    help="""The magnification value to end optimization at.""")

args = parser.parse_args()


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
        detector = correct_distortion(detector, -1.5e-5)

    if not os.path.isdir("./results"):
        os.mkdir("./results")

    iterations = args.iterations
    start_mag = args.starting_magnification
    end_mag = args.ending_magnification
    lowest_noise = np.inf
    best_magnification = None
    best_reconstruction = None
    for j in np.linspace(start_mag, end_mag, iterations):
        print("Magnification:", j)
        source = reconstruct(disk, detector, disk_magnification=j)
        source = np.real(source)
        source += np.abs(np.min(source))
        source /= np.average(source)
        source /= np.max(source)

        patch = source[1500:2500, 1500:2500]
        p_bottom, p_top = np.percentile(patch, (15, 85))
        noise = p_top - p_bottom

        print("Min", p_bottom, "Max", p_top, "Range", noise)
        if noise < lowest_noise:
            lowest_noise = noise
            best_magnification = j
            best_reconstruction = source
            print("*** New Best: ***", "M:", best_magnification, "N:", lowest_noise)
    
    print("Best Magnification", best_magnification)
    io.imsave("./results/optimized_reconstruction.png", img_as_ubyte(best_reconstruction), check_contrast=False)