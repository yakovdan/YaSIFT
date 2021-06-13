import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def read_images():
    img1 = cv2.imread('img1.jpg')
    img2 = cv2.imread('img2.tif')
    return img1, img2


def build_base_image(input_image, input_blur_sigma, output_sigma, min_sigma_diff=0.01):
    """
    Compute the base image in SIFT's image pyramid
    This is done by up scaling 2x and blurring with a Gaussian kernel with a standard
    deviation of sigma.


    input_image is the input image

    input_blur_sigma is the assumed blur of the input image. Note that standard deviations sums quadratically.
    if the added blur is less than min_sigma_diff, just upscale.

    output_sigma is the total blur of the base image.

    min_sigma_diff is the minmal additional blur to be applied that requires a gaussian blur to be applied

    """
    logger('Build base image')
    sigma_diff = (output_sigma ** 2 - input_blur_sigma ** 2)**0.5
    resized_image = cv2.resize(input_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    if sigma_diff >= min_sigma_diff:
        base_image = cv2.GaussianBlur(resized_image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
    else:
        base_image = resized_image
    return base_image

def calculateAmountOfOctaves(image_shape):
    """
    :param image_shape: the width and height of the base image
    :return: how many times the image is to be halved until the smallest dimension is 1
    """
    min_dim = min(image_shape)
    operations = np.log(min_dim) / np.log(2)
    return int(operations)

if __name__ == '__main__':
    a, b = read_images()
    c = cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('test', c)

    cv2.waitKey(0)


