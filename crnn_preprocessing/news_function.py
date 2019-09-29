#coding=utf-8
from .clean_image import projection_x
import numpy as np
import cv2 as cv
import sys

np.set_printoptions(threshold=sys.maxsize)


# 分离滚动字幕前的固定字幕
def news_procress(img, original_img):
    row, col = img.shape

    # p_list = project_y(img)
    # max_num = max(p_list)
    # max_num_index = p_list.index(max_num)

    p_list_x = projection_x(img)
    max_num_x = max(p_list_x)
    max_num_x_index = p_list_x.index(max_num_x)

    i = max_num_x_index
    j = 0
    success = True
    while(success):
        j = j+1
        if img[0][i] == 255 and img[1][i] == 255 and img[row-1][i] == 255 and img[row-2][i] == 255:
            success = False
            line_num = i
        if p_list_x[i] > row-(row * 0.1):
            success = False
            line_num = i
        else:
            p_list_x[i] = 0

        i = p_list_x.index(max(p_list_x))

        if j>col:
            success = False
            print('There is no line between two boxes!')
            line_num = 0
    if line_num != 0:
        img1_0, img1_1 = separate_picture(original_img, line_num)
        # cv.imwrite('2203.jpg', img1_0)
        # cv.imwrite('2204.jpg', img1_1)
        # img2_0 = clean_left_picture(original_img, line_num)
        return img1_0, line_num

    # print line_num
    # return img1_0, img1_1, line_num
    return img, line_num


# 分离为两个图片
def separate_picture(img, separate_line_num):
    row, col = img.shape[0], img.shape[1]
    roi_false = img[0:row-1, 0:separate_line_num]
    roi_true = img[0:row-1, separate_line_num:col-1]
    cv.imshow('roi1', roi_false)
    cv.imshow('roi2', roi_true)
    if cv.waitKey(0) == ord('q'):
        pass
    return roi_false , roi_true


# 将左边的图片删除
def clean_left_picture(img, separate_line_num):
    row, col = img.shape[0], img.shape[1]
    i = 0
    j = 0
    while i < row:
        j = 0
        while j < separate_line_num:
            img[i][j] = 0
            j = j+1
        i = i + 1

    # cv.imshow('clean_right_picture', img)
    # if cv.waitKey(0) == ord('q'):
    #     pass  

    return img


# if __name__ == '__main__':
#     pass


