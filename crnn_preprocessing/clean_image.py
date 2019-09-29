# coding=utf-8
import numpy as np
import sys
import imutils

np.set_printoptions(threshold=sys.maxsize)


# 将图片投影到y轴
def projection_y(img):
    row = img.shape[0]
    col = img.shape[1]
    list_row = []
    for i in range(row):
        white_point_num = 0
        for j in range(col):
            if img[i, j] == 255:
                white_point_num += 1
        list_row.append(white_point_num)

    return list_row


# 将图片投影到x轴
def projection_x(img):
    row = img.shape[0]
    col = img.shape[1]
    list_col = []
    for j in range(col):
        white_point_num = 0
        for i in range(row):
            if img[i, j] == 255:
                white_point_num += 1
        list_col.append(white_point_num)

    return list_col


# 清除噪声
def clean(data, img, type_xy):
    mean = np.mean(data)
    clean_stop_row = -1
    if type_xy == 'x':  # 左右两边的清除标准更严格？
        mean += 1
    for i, element in enumerate(data):
        if element > mean:
            clean_stop_row = i
            break
    if clean_stop_row > 3:   # 小于等于3的为什么不清除
        for j in range(clean_stop_row - 4, -1, -1):
            delate_img(img, j)

    return clean_stop_row


# 清除canny图片的第几行
def delate_img(img, i):
    col = img.shape[1]  # 列数
    for j in range(col):
        img[i, j] = 0


def wash_canny_picture(img):
    data_y = projection_y(img)
    # data_x = projection_x(img)

    top = clean(data_y, img, 'y')   # top

    # img = imutils.rotate_bound(img, 90)  # right
    # clean(data_x, img, 'x')

    img = imutils.rotate_bound(img, 180)  # bottom
    data_y.reverse()
    bottom = clean(data_y, img, 'y')

    # img = imutils.rotate_bound(img, 90)  # left
    # data_x.reverse()
    # clean(data_x, img, 'x')

    img = imutils.rotate_bound(img, 180)  # 恢复

    subtitle_height = len(data_y) - top - bottom
    if top < 0 and bottom < 0:
        subtitle_height = 0

    return img, subtitle_height


# if __name__ == '__main__':
    # img = cv.imread('canny_image.jpg')
    # img, num = wash_canny_picture(img)

    # plt.bar(range(len(data)), data)
    # plt.show()
