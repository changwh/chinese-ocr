# coding=utf-8
import numpy as np
import cv2 as cv
import os
import natsort
import sys

# np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt


# 将图片投影到y轴
def project_y(img):
    row = img.shape[0]
    col = img.shape[1]
    # print row
    i = 0
    j = 0
    list_col = list(range(row))
    while (i < row):
        j = 0
        list_col[i] = 0
        while (j < col):
            if img[i][j] == 255:
                list_col[i] = list_col[i] + 1
            j = j + 1

        # print list_col[i]
        i = i + 1
        # plt.bar(range(len(data)), data)
    # plt.show()

    return list_col


# 清除噪声
def clean(data, img, type_xy):
    mean = np.mean(data)
    if type_xy == 'x':
        mean = mean + 1
    for i, element in enumerate(data):
        if element > mean:
            index = i
            break
    if i > 3:
        j = i - 4
        while (j >= 0):
            delate_img(img, j)
            j = j - 1


# 清除canny图片的第几行
def delate_img(img, i):
    col = img.shape[1]  # 列数
    j = 0
    while (j < col):
        img[i][j] = 0
        j = j + 1


def wash_canny_picture(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    data_y = project_y(img)
    data_x = project_x(img)

    clean(data_y, img, 'y')  # top

    img = rotate_bound(img, 90)  # right
    # clean(data_x, img, 'x')

    img = rotate_bound(img, 90)  # bottom
    data_y.reverse()
    clean(data_y, img, 'y')

    img = rotate_bound(img, 90)  # left
    # data_x.reverse()
    # clean(data_x, img, 'x')

    img = rotate_bound(img, 90)  # 恢复

    # cv.imshow('ss', img)
    # if cv.waitKey(0) == ord('q'):
    #     pass

    return img


# 将图片投影到x轴
def project_x(img):
    row = img.shape[1]
    col = img.shape[0]
    i = 0
    j = 0
    list_col = list(range(row))
    while (i < row):
        j = 0
        list_col[i] = 0
        while (j < col):
            if img[j][i] == 255:
                list_col[i] = list_col[i] + 1
            j = j + 1

        # print list_col[i]
        i = i + 1
        # plt.bar(range(len(data)), data)
    # plt.show()
    return list_col


# 旋转
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv.warpAffine(image, M, (nW, nH))


if __name__ == '__main__':
    img = cv.imread('canny_image.jpg')
    wash_canny_picture(img)

    # plt.bar(range(len(data)), data)
    # plt.show()
