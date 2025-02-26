import numpy as np
import cv2 as cv

from pathlib import Path


def harris_method(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.06)

    dst = cv.dilate(dst, None)

    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    return image


def improved_harris(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, 300, 0.06, 4)
    corners = np.intp(corners)

    for i in corners:
        x, y = i.ravel()
        cv.circle(image, (x, y), 3, 255, -1)

    return image


def sift_method(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kps = sift.detect(gray, None)

    for kp in kps:
        (x, y) = kp.pt
        cv.circle(image, (round(x), round(y)), 3, (0, 255, 0), -1)

    return image


if __name__ == '__main__':
    folder = Path('images')
    output_folder = Path('output')
    # filename = "20151101_125056.jpg"

    for file in folder.iterdir():

        img = cv.imread(str(file))

        # moravec_method(img.copy())
        harris = harris_method(img.copy())
        cv.imwrite(str(output_folder / ("harris_" + file.name)), harris)

        shi = improved_harris(img.copy())
        cv.imwrite(str(output_folder / ("shi_tomasi_" + file.name)), shi)

        sift = sift_method(img.copy())
        cv.imwrite(str(output_folder / ("sift_" + file.name)), sift)
