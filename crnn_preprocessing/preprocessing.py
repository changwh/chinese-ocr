# coding=utf-8
import numpy as np
import cv2 as cv
import sys
import os
from . import clean_image

# import natsort

np.set_printoptions(threshold=sys.maxsize)


# np.set_printoptions(threshold=np.nan)
# from matplotlib import pyplot as plt


def dilate_demo(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # kernel(5, 5)
    dst = cv.dilate(thresh, kernel)
    # cv.imshow('dilate image', dst)
    return dst


def precess(thresh1, canny_img):
    row = thresh1.shape[0]
    col = thresh1.shape[1]
    # print(row, col)
    i = 0
    j = 0
    while (i < row):
        j = 0
        while (j < col):
            if (thresh1[i, j] == 255) and (canny_img[i, j, 0] == 255):
                canny_img[i, j] = [255, 255, 255]
            else:
                canny_img[i, j] = [0, 0, 0]
            j = j + 1
        i = i + 1
    return canny_img


def precess_2(img):
    row = img.shape[0]
    col = img.shape[1]
    i = 0
    j = 0
    while (j < col):
        if img[0, j, 0] == 255:
            # global line_extent
            # line_extent = 0
            clean_the_line(img, 1, j, row, col)
            # print(line_extent)
        j = j + 1

    i = 0
    j = 0
    while (j < col):
        if img[row - 1, j, 0] == 255:
            # global line_extent
            # line_extent = 0
            clean_the_line(img, row - 2, j, row, col)
            # print(line_extent)
        j = j + 1
    i = 0
    j = 0
    while (i < row):
        if img[i, j, 0] == 255:
            # global line_extent
            # line_extent = 0
            clean_the_line(img, i, j, row, col)
            # print(line_extent)
        i = i + 1

    return img


def clean_the_line(img, i, j, row, col):
    img[i, j] = [0, 0, 0]
    # global line_extent
    # line_extent = line_extent + 1
    # if line_extent>100:
    #   return
    # print(i, j)
    if img[i, j + 1, 0] == 255 and j != col - 2:  # 右边
        clean_the_line(img, i, j + 1, row, col)
    if img[i + 1, j + 1, 0] == 255 and j != col - 2 and i != row - 2:  # 右下
        clean_the_line(img, i + 1, j + 1, row, col)
    if img[i + 1, j, 0] == 255 and i != row - 2:  # 下边
        clean_the_line(img, i + 1, j, row, col)
    if img[i + 1, j - 1, 0] == 255 and i != row - 2 and j != 1:  # 左下
        clean_the_line(img, i + 1, j - 1, row, col)
    if img[i, j - 1, 0] == 255 and j != 1:  # 左边
        clean_the_line(img, i, j - 1, row, col)
    if img[i - 1, j + 1, 0] == 255 and j != col - 2 and i != 1:  # 右上
        clean_the_line(img, i - 1, j + 1, row, col)
    if img[i - 1, j, 0] == 255 and i != 1:  # 上边
        clean_the_line(img, i - 1, j, row, col)
    if img[i - 1, j - 1, 0] == 255 and i != 1 and j != 1:  # 左上
        clean_the_line(img, i - 1, j - 1, row, col)
    return 0


def dilate_demo2(img):
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow('binary iamge', thresh)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))  # kernel(8, 8)
    dst = cv.dilate(thresh, kernel)
    # cv.imwrite('pengzhang.png', dst)
    # cv.imshow('dilate image', dst)
    return dst


def erode_demo(img):
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # kernel(5,5)
    dst = cv.erode(thresh, kernel=kernel)
    # cv.imwrite('jieguo.png', dst)
    # cv.imshow('erode_demo', dst)
    return dst


def get_result(step2, binary_img):
    row = binary_img.shape[0]
    col = binary_img.shape[1]
    i = 0
    j = 0
    while (i < row):
        j = 0
        while (j < col):
            if step2[i, j] == 0:
                binary_img[i, j] = [0, 0, 0]
            j = j + 1
        i = i + 1
    return binary_img


def fit(i, j, img):
    ff = 200
    img[i, j, 2] = ff
    img[i, j + 1, 2] = ff
    img[i, j - 1, 2] = ff
    img[i + 1, j, 2] = ff
    img[i - 1, j, 2] = ff
    img[i + 1, j + 1, 2] = ff
    img[i - 1, j - 1, 2] = ff
    img[i + 1, j - 1, 2] = ff
    img[i - 1, j + 1, 2] = ff


def convert_demo(image):
    height, width, channels = image.shape
    print("width:%s,height:%s,channels:%s" % (width, height, channels))

    for row in range(height):
        for list in range(width):
            for c in range(channels):
                pv = image[row, list, c]
                image[row, list, c] = 255 - pv
    # cv.imshow("AfterDeal", image)


def convert(image):
    # image = cv.imread(image,0)
    height = image.shape[0]
    width = image.shape[1]
    for i in range(height):
        for j in range(width):
            image[i, j] = 255 - image[i, j]
    return image


def main(left, top, right, bottom, img, videoName, outputPath, frameNum, index):
    # left = 553
    # top = 615
    # right = 737
    # bottom = 673
    # picture_name = '2'
    base_name = videoName.split('/')[-1]

    roi = img[top:bottom, left:right]

    # 读取灰度图
    gray_image = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # 边缘检测得到边缘图片
    canny_img = cv.Canny(roi, 40, 120)
    # cv.imshow('canny 1', canny_img)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "1_canny_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img)

    # 阈值处理
    ret, thresh1 = cv.threshold(gray_image, 225, 255, cv.THRESH_BINARY)
    # cv.imshow('thresh1!', thresh1)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "2_gray_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), thresh1)

    # 膨胀
    thresh2 = dilate_demo(thresh1)
    # cv.imshow('dilate_demo', thresh2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "3_dilate_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), thresh2)

    canny_img = cv.cvtColor(canny_img, cv.COLOR_GRAY2BGR)
    canny_img2 = precess(thresh2, canny_img)
    canny_img2, subtitle_height = clean_image.wash_canny_picture(canny_img2)
    # canny_img2 = precess_2(canny_img2)
    # cv.imshow('canny 2', canny_img2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)
    # cv.imwrite(os.path.join(outputPath, "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)
    # np.savetxt(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
    #                         "4_canny2_{}_{}_{}.txt".format(base_name.split('.')[0], frameNum, index)), canny_img2, fmt='%d')

    step1 = dilate_demo2(canny_img2)
    step2 = erode_demo(step1)

    output = get_result(step2, roi)
    convert(output)

    img[top:bottom, left:right] = output

    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "5_erode_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), step2)

    # cv.imshow('final!!!!!', img)
    # if cv.waitKey(0) == ord('q'):
    #     pass

    cv.imwrite(os.path.join(outputPath, "final_{}_{}.jpg".format(base_name.split('.')[0], str(frameNum))), img)
    return img, subtitle_height


def p_picture(text_recs, img, videoName, outputPath, frameNum):
    subtitle_height_list = []
    for index in range(len(text_recs)):
        left = max(min(text_recs[index][0], text_recs[index][4]), 0)
        top = min(text_recs[index][1], text_recs[index][3])
        right = min(max(text_recs[index][2], text_recs[index][6]), img.shape[1])
        bottom = max(text_recs[index][5], text_recs[index][7])
        img, subtitle_height = main(left, top, right, bottom, img, videoName, outputPath, frameNum, index)
        subtitle_height_list.append(subtitle_height)
    return img, subtitle_height_list


if __name__ == '__main__':
    pass
