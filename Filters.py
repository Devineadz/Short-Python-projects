# Diana Zitting-Rioux
#
# February 2, 2020

import numpy as np
import cv2
import math

# path, image from https://www.wikiart.org/en/vincent-van-gogh/houses-in-auvers-2-1890
path = r'houses-in-auvers-2-1890.jpg'

# Original image
image = cv2.imread(path)

# Show image in original size
cv2.imshow('Original', image)
cv2.waitKey(0)


# resize function
def resize(original_image, n_factor):
    # create a new black image with the size of original image / n_factor
    new_width = int(image.shape[0] / n_factor)
    new_height = int(image.shape[1] / n_factor)
    new_arr = np.zeros([new_width, new_height, 3], dtype=np.uint8)
    # loop through the original image and copy every nth pixel
    for x in range(0, image.shape[0]-1):
        for y in range(0, image.shape[1]-1):
            if x % n_factor == 0 and y % n_factor == 0:
                x_new = int(x/n_factor)-1
                y_new = int(y/n_factor)-1
                new_arr[x_new, y_new] = image[x, y]
    return new_arr


image2 = resize(image, 2)
cv2.imshow('size 2', image2)
cv2.waitKey(0)

image4 = resize(image, 4)
cv2.imshow('size 1/4', image4)
cv2.waitKey(0)

image8 = resize(image, 8)
cv2.imshow('size 1/8', image8)
cv2.waitKey(0)

image16 = resize(image, 16)
cv2.imshow('size 1/16', image16)
cv2.waitKey(0)

nearest16 = cv2.resize(image16, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
cv2.imshow('nearest neighbor', nearest16)
cv2.waitKey(0)

bicubic16 = cv2.resize(image16, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Bicubic interpolation', bicubic16)
cv2.waitKey(0)

linear16 = cv2.resize(image16, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
cv2.imshow('linear interpolation', linear16)
cv2.waitKey(0)

# Question 2 a Shifting to top right corner
shifted_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
# Filter to shift to top right corner
mat1 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float)
shifted_img = cv2.filter2D(image, -1, mat1)

cv2.imshow('Shifted img', shifted_img)
cv2.waitKey(0)


# Question 2b. Gaussian filter
def gaussian(filter_size, img, delta):
    # Check if kernel size is an odd positive number
    if (filter_size % 2) != 1 or filter_size < 0:
        print("Kernel size must be an odd, positive number")
        return
    new_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    gauss_kernel = np.zeros((filter_size, filter_size), dtype=np.float)
    for k in range(0, int(filter_size/2)+1):
        for m in range(0, int(filter_size/2)+1):
            # Set distance from the middle pixel
            x_dist = int(filter_size/2)-k
            y_dist = int(filter_size/2)-m
            # Gaussian function
            gaussian_val = 1/(2*math.pi*delta**2)*math.e**(-(x_dist**2+y_dist**2)/(2*delta**2))
            gauss_kernel[k, m] = gaussian_val
            # Copy result to the other pixels with same distance
            if k != int(filter_size/2) and m != int(filter_size/2):
                gauss_kernel[k, filter_size-1-m] = gaussian_val
                gauss_kernel[filter_size-1-k, m] = gaussian_val
                gauss_kernel[filter_size-1-k, filter_size-1 - m] = gaussian_val
            # Copy result to other pixels with same distance (vertical to middle pixel)
            if k != int(filter_size/2) and m == int(filter_size/2):
                gauss_kernel[m, k] = gaussian_val
                gauss_kernel[m, filter_size-1-k] = gaussian_val
                gauss_kernel[filter_size-1-k, m] = gaussian_val
    new_img = cv2.filter2D(img, -1, gauss_kernel)
    return new_img


gaussian_img3 = gaussian(3, image, 1)
cv2.imshow('Kernel size 3', gaussian_img3)
cv2.waitKey(0)
gaussian_img5 = gaussian(5, image, 1)
cv2.imshow('Kernel size 5', gaussian_img5)
cv2.waitKey(0)


# Getting the picture in gray scale
img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
r, c = img_gray.shape


# Question 2c. Take input of two variances and subtract the filtered images
def subtracted_gaussian(kernel_size1, kernel_size2):
    # calculate filtered images
    gaussian1 = gaussian(kernel_size1, img_gray, 5)
    gaussian2 = gaussian(kernel_size2, img_gray, 3)
    combined_img = np.subtract(gaussian1, gaussian2)
    return combined_img


sub_gaussian_img = subtracted_gaussian(3, 3)
cv2.imshow('subtracted img', sub_gaussian_img)
cv2.waitKey(0)


# Question 3a. Sobel filter for Sx and Sy
# Sobel filter for Sx
def sobel_sx():
    mat2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float)
    # normalize filter
    mat2 = np.multiply(mat2, 1/8)
    sobelimg_sx = cv2.filter2D(img_gray, -1, mat2)
    return sobelimg_sx


# Sobel filter for Sy
def sobel_sy():
    mat3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)
    # normalize filter
    mat3 = np.multiply(mat3, 1/8)
    sobelimg_sy = cv2.filter2D(img_gray, -1, mat3)
    return sobelimg_sy


sobel_x = sobel_sx()
cv2.imshow('Sobel Sx', sobel_x)

sobel_y = sobel_sy()
cv2.imshow('Sobel Sy', sobel_y)
cv2.waitKey(0)


#Questiob 3b.
def orientation_map():
    or_map = np.zeros([r, c, 3], dtype=np.uint8)
    for n in range(0, r):
        for p in range(0, c):
            or_map[n, p] = [0, sobel_y[n, p], sobel_x[n, p]]
    return or_map


ori_map = orientation_map()
cv2.imshow('Orientation map', ori_map)
cv2.waitKey(0)

#Question 3c.
def magnitude():
    mag_img = np.zeros((r, c), dtype=np.uint8)
    for o in range(0, r):
        for p in range(0, c):
            # calculate gradients magnitude at (o, p)
            mag_img[o, p] = math.sqrt(sobel_x[o, p]**2+sobel_y[o, p]**2)
    return mag_img

mag_pic = magnitude()
cv2.imshow('Magnitude', mag_pic)
cv2.waitKey(0)

# Question 3d. Canny function
canny_pic = cv2.Canny(mag_pic, 0, 100)
cv2.imshow('Canny', canny_pic)
cv2.waitKey(0)

# Close all open windows
cv2.destroyAllWindows()

