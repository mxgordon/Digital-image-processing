import cv2 as cv
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


def read_and_convert_img(path):
    im = cv.imread(str(path))
    return cv.cvtColor(im, cv.COLOR_RGB2BGR)


def mse(img1: np.ndarray, img2: np.ndarray):
    diff = img1 - img2
    diff **= 2

    return diff.mean()


# https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#:%7E:text=Contrast%20Limited%20AHE%20(CLAHE)%20is,slope%20of%20the%20transformation%20function.
def contrast_limited_adaptive_histo_eq(image):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(2, 2))
    cl = clahe.apply(l_channel)
    limg = cv.merge((cl, a, b))

    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)


def goofy_power_law(image):
    c = 3
    gamma = 1.6

    new_image = cv.pow(image/255, gamma) * c
    normalized_image = (255 * new_image).clip(0, 255)
    return normalized_image.astype(np.uint8)


def thresh_and_stretch_white(image, thresh=150):
    image = image.copy().astype(np.float32)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = (gray > thresh)

    image = (image - 40) * 2.4
    image = image.clip(0, 255).astype(np.uint8)
    image[mask] = (255, 255, 255)
    return image



def show_both(bad, better):
    result = np.hstack((cv.cvtColor(bad, cv.COLOR_BGR2RGB), cv.cvtColor(better, cv.COLOR_BGR2RGB)))
    cv.imshow('Result', result)
    cv.waitKey(0)



def show_histos(test, bad, truth):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    ax1: plt.Axes

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        # print(np.log(img).astype(np.uint8), img.dtype)
        histr = cv.calcHist([test], [i], None, [256], [0, 256])
        # print(histr, histr.dtype)
        histr[histr == 0] = 0.00001
        histr = np.log(histr)#,where=histr>0)
#         print(histr, histr.dtype)
        ax1.plot(histr, color=col)
        ax1.set_xlim([0, 256])
        ax1.set_title("Foggy Image")


    for i, col in enumerate(color):
        histr = cv.calcHist([bad], [i], None, [256], [0, 256])
#         print(histr, histr.dtype)
        histr[histr == 0] = 0.00001
        histr = np.log(histr)#,where=histr>0)
#         print(histr, histr.dtype)
        ax2.plot(histr, color=col)
        ax2.set_xlim([0, 256])
        ax2.set_title("Ground Truth")


    for i, col in enumerate(color):
        histr = cv.calcHist([truth], [i], None, [256], [0, 256])
#         print(histr, histr.dtype)
        histr[histr == 0] = 0.00001
        histr = np.log(histr)#,where=histr>0)
#         print(histr, histr.dtype)
        ax3.plot(histr, color=col)
        ax3.set_xlim([0, 256])
        ax3.set_title("Method 3")

    plt.show(block=True)


if __name__ == '__main__':
    img_folder = Path("./images")
    matplotlib.use('TkAgg')

    truth_img_path = img_folder/"20151103_144133.jpg"
    bad_img_path = img_folder/"20151102_074127.jpg"

    truth_img = read_and_convert_img(truth_img_path)
    bad_img = read_and_convert_img(bad_img_path)

    method1 = goofy_power_law(bad_img)
    method2 = thresh_and_stretch_white(bad_img)
    method3 = contrast_limited_adaptive_histo_eq(bad_img)
    # method4 = thresh_and_stretch_white(method1.copy(), thresh=180)

    # show_histos()

    print(f"Starting error: {mse(bad_img, truth_img):2.2f}")
    print(f"Method 1 error: {mse(method1, truth_img):2.2f}")
    print(f"Method 2 error: {mse(method2, truth_img):2.2f}")
    print(f"Method 3 error: {mse(method3, truth_img):2.2f}")
    # print(f"Method 4 error: {mse(method4, truth_img):2.2f}")

    cv.imwrite("method1.jpg", cv.cvtColor(method1, cv.COLOR_RGB2BGR))
    cv.imwrite("method2.jpg", cv.cvtColor(method2, cv.COLOR_RGB2BGR))
    cv.imwrite("method3.jpg", cv.cvtColor(method3, cv.COLOR_RGB2BGR))

    # cv.imshow()
    # show_both(method3, truth_img)
    show_histos(bad_img, truth_img, method3)

    # plt.imshow(method1, block=True)
    # plt.show()
    # print(mse()

