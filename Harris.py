# Diana Zitting-Rioux, 40017023
# COMP 425, assignment 2
# February 2, 2020

import numpy as np
import cv2

# path
path = r'yosemite\Yosemite2.jpg'
path2 = r'yosemite\Yosemite1.jpg'

# Sobel filter for Sx
def sobel_sx(img_gray):
    mat2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    # normalize filter
    mat2 = np.multiply(mat2, 1/8)
    sobelimg_sx = cv2.filter2D(img_gray, -1, mat2)
    return sobelimg_sx


# Sobel filter for Sy
def sobel_sy(img_gray):
    mat3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    # normalize filter
    mat3 = np.multiply(mat3, 1/8)
    sobelimg_sy = cv2.filter2D(img_gray, -1, mat3)
    return sobelimg_sy


def harris(img_path):
    image_original = cv2.imread(img_path)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # compute gradients Ix, Iy
    x_sobel = sobel_sx(img_gray)
    y_sobel = sobel_sy(img_gray)
    # Display both gradients
    cv2.imshow('Sobel Sx', x_sobel)
    cv2.imshow('Sobel Sy', y_sobel)
    cv2.waitKey(0)

    # Calculate Ixx
    x_xmatrix = np.square(x_sobel)
    x_xblurred = cv2.GaussianBlur(x_xmatrix, (3, 3), 1)
    cv2.imshow('Gradient x*x', x_xblurred)
    cv2.waitKey(0)

    # Calculate Iyy
    y_ymatrix = np.square(y_sobel)
    y_yblurred = cv2.GaussianBlur(y_ymatrix, (3, 3), 1)
    cv2.imshow('Gradient y*y', y_yblurred)
    cv2.waitKey(0)

    # Calculate Ixy
    x_ymatrix = np.multiply(x_sobel, y_sobel)
    x_yblurred = cv2.GaussianBlur(x_ymatrix, (3, 3), 1)
    cv2.imshow('Gradient x*y', x_yblurred)
    cv2.waitKey(0)

    # alpha needs to be between 0.04 and 0.06
    alpha = 0.05

    # R(H) response, slide a 5x5 window over the matrix, check if pixel value is over 100 before calculating and calculate r for each proper pixel
    img_width, img_height = img_gray.shape
    r_img = np.zeros((img_width-2, img_height-1), dtype = np.uint8)
    for i in range(2, img_width-2):
        for j in range(2, img_height-2):
            if x_yblurred[i, j]>100:
                Hxx = np.sum(x_xblurred[i - 2:i + 3, j - 2:j + 3])
                Hyy = np.sum(y_yblurred[i - 2:i + 3, j - 2:j + 3])
                Hxy = np.sum(x_yblurred[i - 2:i + 3, j - 2:j + 3])
                det = Hxx * Hyy - Hxy**2
                trace = Hxx + Hyy
                r = det - alpha * trace**2
                r_img[i][j] = r
            else:
                r_img[i][j] = 0
    cv2.imshow('r', r_img)
    cv2.waitKey(0)

    r_width, r_length = r_img.shape
    r_keypoints = []

    # Suppress to 0 if neighbors have a larger value
    for x in range(r_width-1):
        for y in range(r_length-1):
            if r_img[x][y]<r_img[x-1][y-1] or r_img[x][y]<r_img[x-1][y] or r_img[x][y]<r_img[x-1][y+1] or r_img[x][y]<r_img[x][y-1] or r_img[x][y]<r_img[x][y+1] or r_img[x][y]<r_img[x+1][y-1] or r_img[x][y]<r_img[x+1][y] or r_img[x][y]<r_img[x+1][y+1]:
                r_img[x][y] = 0

    threshold = 0.8 * r_img.max()
    # Threshold r with 80% of image maximum pixel value
    for x in range(r_width):
        for y in range(r_length):
            if r_img[x][y] > threshold:
                r_keypoints.append(cv2.KeyPoint(y, x, 1))
    cv2.drawKeypoints(image_original, r_keypoints, image_original, color=(255, 0, 0))
    cv2.imshow('KeyPoints', image_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


harris(path)
harris(path2)