#coding=utf-8
import cv2 as cv
import numpy as np


# 通过量化降低来计算 kernel = 86， 3个区间
def conculate_proportion_v3(img, out_name='the_whole_test'):

    row = img.shape[0]
    col = img.shape[1]
    row_start = row // 8
    row_end = row_start * 7
    col_start = col // 8
    col_end = col_start * 7

    the_list = np.arange(27)

    img = reduce_colors(img, 86)

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            x = img[i, j, 0]
            y = img[i, j, 1]
            z = img[i, j, 2]
            index = x * 9 + y * 3 + z
            the_list[index] += 1

    is_background = False

    # 找出前三个最大元素的下标
    list2 = the_list.argsort()[-4:][::-1]

    # Ture:有底色
    if the_list[list2[1]] * 3 > the_list[list2[0]] * 2:
        is_background = True
    elif the_list[list2[2]] * 3 < the_list[list2[1]] * 2:
        is_background = True
    elif the_list[list2[3]] * 3 > the_list[list2[2]] * 2:
        is_background = True

    if is_background == True:
        return 0, img
    else:
        if list2[0] == 26:
            return 2,img
        thresh1 = keep_the_largest(img, list2[0], out_name)

        gray_image_real = cv.cvtColor(thresh1, cv.COLOR_BGR2GRAY)
        return 1, gray_image_real


def keep_the_largest(img, index, out_name):
    row = img.shape[0]
    col = img.shape[1]

    for i in range(0, row):
        for j in range(0, col):
            x = img[i, j, 0]
            y = img[i, j, 1]
            z = img[i, j, 2]
            index_org = x * 9 + y * 3 + z

            if index_org == index:
                img[i, j] = [255, 255, 255]
            else:
                img[i, j] = [0, 0, 0]

    return img


# 量化降低
def reduce_colors(img, kernel):

    img = img // kernel

    return img
