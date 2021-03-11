# Diana Zitting-Rioux, 40017023
# COMP 425, assignment 2
# February 2, 2020

import numpy as np
import cv2
import math

# path
path = r'hough1.png'
path2 = r'hough2.png'



def hough(img_path):
    # Original image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    width, height = image.shape

    d = round(math.sqrt(width**2+height**2))
    thetas = np.arange(0, 180, 1)

    # To get cosine and sine values in radians
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    # Create accumulator
    accumulator = np.zeros((d*2, 180), dtype=np.uint8)


    # Convert with Canny function
    edges = cv2.Canny(image, 50, 200)
    for x in range (width):
        for y in range (height):
            if edges[x][y] > 0:
                for theta in range(0, 180):
                    rho = round(x*cos_thetas[theta]+ y* sin_thetas[theta])
                    accumulator[rho, theta] +=1
    cv2.imshow('Accumulator', accumulator)
    cv2.waitKey(0)

    return accumulator


# function to create lines
def hough_lines(accumulator, threshold, img_path):
    accumulator_width, accumulator_height = accumulator.shape
    # define line color and thickness
    color = (0, 255, 0)
    thickness = 2
    color_img = cv2.imread(img_path)
    height = color_img.shape[0]
    width = color_img.shape[1]
    d = round(math.sqrt(width**2+height**2))
    thetas = np.arange(0, 180, 1)
    rhos = np.arange(0, 2 * d, 1)

    # check if value in i, j is above the treshold and convert it to cartesian coordinate lines
    for i in range(accumulator_width):
        for j in range(accumulator_height):
            if accumulator[i][j] > threshold:
                x = np.cos(np.deg2rad(thetas[j]))
                y = np.sin(np.deg2rad(thetas[j]))
                x0 = (x * rhos[i])
                y0 = (y * rhos[i])
                x1 = int(x0 + height *2* (-y))
                x2 = int(x0 - height *2* (-y))
                y1 = int(y0 + width *2* (x))
                y2 = int(y0 - width *2* (x))
                color_img = cv2.line(color_img, (y1, x1), (y2, x2), color, thickness)
    cv2.imshow('Test image', color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


accumulator = hough(path)
hough_lines(accumulator, 40, path)
accumulator2 = hough(path2)
hough_lines(accumulator2, 82, path2)