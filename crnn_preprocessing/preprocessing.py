# coding=utf-8
import numpy as np
import cv2 as cv
import sys
import os
import random
from . import clean_image
from .v2.final import main as main_v2

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


def dilate_demo2(img, dilate_kernel):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow('binary iamge', thresh)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_kernel, dilate_kernel))    # kernel(8, 8)
    dst = cv.dilate(thresh, kernel)
    # cv.imwrite('pengzhang.png', dst)
    # cv.imshow('dilate image', dst)
    return dst


def erode_demo(img, erode_kernel):
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (erode_kernel, erode_kernel))      # kernel(5,5)
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


def get_the_width(gray_image, canny_img2):
    ret, thresh1 = cv.threshold(gray_image, 245, 255, cv.THRESH_BINARY)
    # cv.imshow('1', thresh1)
    # cv.imwrite('2.jpg', canny_img2)
    row = thresh1.shape[0]
    col = thresh1.shape[1]
    i = 0
    j = 0
    num = 0
    list1 = []
    while (i < row):
        j = 0
        while(j < col):
            if thresh1[i, j] == 255:
                num = num + 1
                list1.append([i, j])
            j = j+1
        i = i+1
    v = 0

    index = 0
    if num > 0:
        the_width_list = np.arange(50)
        while v < 50 and index < 300:
            the_width = creat_random(canny_img2, num, list1)
            if the_width != 0:
                the_width_list[v] = the_width
                v = v+1
            index = index + 1
        counts = np.bincount(the_width_list)
        final_width = np.argmax(counts)
        # print(final_width)
        return final_width
    return 0


def creat_random(canny_img2, num, list1):
    ss = random.randint(0, num-1)
    max_x = canny_img2.shape[0]
    max_y = canny_img2.shape[1]

    x = list1[ss][0]
    y = list1[ss][1]
    top = 0
    top_index = 0
    if canny_img2[x, y, 0] == 0:
        while(top_index == 0):
            x = x+1
            if x < max_x:
                if canny_img2[x, y, 0] == 0:
                    top = top+1
                else:
                    top_index = 1
            else:
                top_index = 1

    x = list1[ss][0]
    y = list1[ss][1]
    bottom = 0
    bottom_index = 0
    if canny_img2[x, y, 0] == 0:
        while(bottom_index == 0):
            x = x-1
            if x >= 0:
                if canny_img2[x, y, 0] == 0:
                    bottom = bottom+1
                else:
                    bottom_index = 1
            else:
                bottom_index = 1

    x = list1[ss][0]
    y = list1[ss][1]
    left = 0
    left_index = 0
    if canny_img2[x, y, 0] == 0:
        while(left_index == 0):
            y = y-1
            if y >= 0:
                if canny_img2[x, y, 0] == 0:
                    left = left+1
                else:
                    left_index = 1
            else:
                left_index = 1

    x = list1[ss][0]
    y = list1[ss][1]
    right = 0
    right_index = 0
    if canny_img2[x, y, 0] == 0:
        while(right_index == 0):
            y = y + 1
            if y < max_y:
                if canny_img2[x, y, 0] == 0:
                    right = right+1
                else:
                    right_index = 1
            else:
                right_index = 1

    the_num1 = top + bottom
    the_num2 = left + right

    if the_num1 < the_num2:
        the_final_num = the_num1
    else:
        the_final_num = the_num2

    if the_final_num < 10 and the_final_num > 1:
        # print the_final_num + 1
        return the_final_num + 1

    # print top , bottom
    # print left, right

    return 0


def main(left, top, right, bottom, img, videoName, outputPath, frameNum, index):
    # left = 553
    # top = 615
    # right = 737
    # bottom = 673
    # picture_name = '2'
    base_name = videoName.split('/')[-1]

    roi = img[top:bottom, left:right]
    roi_copy = roi.copy()

    # 读取灰度图
    gray_image = cv.cvtColor(roi_copy, cv.COLOR_BGR2GRAY)

    # 边缘检测得到边缘图片
    canny_img = cv.Canny(roi_copy, 40, 120)  # 与视频清晰度相关,清晰度越高,阈值可相应调高(1:3)
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
    the_width = get_the_width(gray_image, canny_img2)
    canny_img2, subtitle_height = clean_image.wash_canny_picture(canny_img2)
    # canny_img2 = precess_2(canny_img2)
    # cv.imshow('canny 2', canny_img2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)
    # cv.imwrite(os.path.join(outputPath, "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)
    # np.savetxt(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
    #                         "4_canny2_{}_{}_{}.txt".format(base_name.split('.')[0], frameNum, index)), canny_img2, fmt='%d')

    # TODO:需要通过测试获得更多的参数
    if the_width >= 6:
        dilate_kernel = 12
        erode_kernel = 10
    if the_width >= 4 and the_width < 6:
        dilate_kernel = 7
        erode_kernel = 5
    elif the_width <= 2:
        dilate_kernel = 4
        erode_kernel = 3
    elif the_width > 2 and the_width < 4:
        dilate_kernel = 6
        erode_kernel = 4

    step1 = dilate_demo2(canny_img2, dilate_kernel)
    step2 = erode_demo(step1, erode_kernel)

    output = get_result(step2, roi_copy)
    convert(output)

    # img[top:bottom, left:right] = output

    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "5_erode_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), step2)

    # cv.imshow('final!!!!!', img)
    # if cv.waitKey(0) == ord('q'):
    #     pass
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "6_output_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), output)
    # cv.imwrite(os.path.join(outputPath, "final_{}_{}.jpg".format(base_name.split('.')[0], str(frameNum))), img)
    return output, subtitle_height, canny_img2


# 判断两个矩形框是否有重叠部分，并输出重叠部分位置
def get_overlap_coordinate(location_a, location_b):
    x = [location_a[0], location_a[2], location_b[0], location_b[2]]
    y = [location_a[1], location_a[5], location_b[1], location_b[5]]
    overlap_location = []
    x.sort()
    y.sort()
    if y[1] != location_a[5]:
        for index, value in enumerate(x):
            if value == location_a[0]:
                if x[index+1] != location_a[2]:
                    overlap_location = [y[1], y[2], x[1], x[2]]
            elif value == location_b[0]:
                if x[index+1] != location_b[2]:
                    overlap_location = [y[1], y[2], x[1], x[2]]
    # if overlap_location:
    #     print(overlap_location)
    # else:
    #     pass
    return overlap_location


def get_union_coordinate(location_a, location_b):
    x = [location_a[0], location_a[2], location_b[0], location_b[2]]
    y = [location_a[1], location_a[5], location_b[1], location_b[5]]
    union_location = []
    x.sort()
    y.sort()
    if y[1] != location_a[5]:
        for index, value in enumerate(x):
            if value == location_a[0]:
                if x[index+1] != location_a[2]:
                    union_location = [y[0], y[3], x[0], x[3]]
            elif value == location_b[0]:
                if x[index+1] != location_b[2]:
                    union_location = [y[0], y[3], x[0], x[3]]

    return union_location


def get_localtions(text_recs, img):
    locations = []
    for index in range(len(text_recs)):
        left = max(min(text_recs[index][0], text_recs[index][4]), 0)
        top = min(text_recs[index][1], text_recs[index][3])
        right = min(max(text_recs[index][2], text_recs[index][6]), img.shape[1])
        bottom = max(text_recs[index][5], text_recs[index][7])
        location = [top, bottom, left, right]
        locations.append(location)
    return locations


def test_get_overlap_coordinate():
    get_overlap_coordinate([600, 750, 900, 1050], [650, 700, 950, 1000])
    get_overlap_coordinate([600, 700, 950, 1050], [650, 750, 900, 1000])
    get_overlap_coordinate([600, 750, 900, 1050], [650, 700, 950, 1000])
    get_overlap_coordinate([600, 750, 950, 1050], [650, 700, 900, 1000])
    get_overlap_coordinate([600, 750, 900, 1000], [650, 700, 950, 1050])
    get_overlap_coordinate([600, 700, 950, 1000], [650, 750, 900, 1050])
    get_overlap_coordinate([600, 700, 900, 1050], [650, 750, 950, 1000])

    get_overlap_coordinate([600, 650, 900, 950], [700, 750, 1000, 1050])


def p_picture(text_recs, is_scroll, img, frameNum, videoName, outputPath, version):
    subtitle_height_list = []
    canny2_img_list = []
    output_list = []

    location_list = get_localtions(text_recs, img)

    for index in range(len(text_recs)):
        top = location_list[index][0]
        bottom = location_list[index][1]
        left = location_list[index][2]
        right = location_list[index][3]

        if version == 1:
            output, subtitle_height, canny2_img = main(left, top, right, bottom,
                                                       img, videoName, outputPath, frameNum, index)
        elif version == 2:
            output, subtitle_height, canny2_img = main_v2(left, top, right, bottom,
                                                       img, videoName, outputPath, frameNum, index)
        output_list.append(output)

        subtitle_height_list.append(subtitle_height)
        canny2_img_list.append(canny2_img)

    overlap_part = []
    overlap_coordinate_list = []

    for index in range(len(text_recs)):
        if index + 1 < len(text_recs):
            # 计算当前文本框与下一文本框是否存在重叠部分，若存在，获取重叠部分坐标
            overlap_coordinate = get_overlap_coordinate(text_recs[index], text_recs[index + 1])
            if overlap_coordinate:
                # 计算重叠部分在当前文本框中的相对位置
                relative_top = overlap_coordinate[0] - text_recs[index][1]
                relative_bottom = overlap_coordinate[1] - text_recs[index][1]
                relative_left = overlap_coordinate[2] - text_recs[index][0]
                relative_right = overlap_coordinate[3] - text_recs[index][0]
                overlap_part_1 = output_list[index][relative_top:relative_bottom, relative_left:relative_right]
                overlap_p1_copy = overlap_part_1.copy()

                # 计算重叠部分在下一文本框中的相对位置
                relative_top = overlap_coordinate[0] - text_recs[index + 1][1]
                relative_bottom = overlap_coordinate[1] - text_recs[index + 1][1]
                relative_left = overlap_coordinate[2] - text_recs[index + 1][0]
                relative_right = overlap_coordinate[3] - text_recs[index + 1][0]
                overlap_part_2 = output_list[index + 1][relative_top:relative_bottom, relative_left:relative_right]
                overlap_p2_copy = overlap_part_2.copy()

                # 将两个文本框的重叠部分中的非白色部分进行整合
                height = relative_bottom - relative_top
                width = relative_right - relative_left
                for i in range(height):
                    for j in range(width):
                        if overlap_p2_copy[i, j].any() != 255:
                            overlap_p1_copy[i, j] = overlap_p2_copy[i, j]

                overlap_coordinate_list.append(overlap_coordinate)
                overlap_part.append(overlap_p1_copy)

        # 将处理结果拼回原图（不考虑重叠问题）
        img[location_list[index][0]:location_list[index][1], location_list[index][2]:location_list[index][3]] = output_list[index]

    # 将整合后的重叠部分拼回原图
    for index in range(len(overlap_part)):
        img[overlap_coordinate_list[index][0]:overlap_coordinate_list[index][1], overlap_coordinate_list[index][2]:overlap_coordinate_list[index][3]] = overlap_part[index]

    cv.imwrite(os.path.join(outputPath, "final_{}_{}.jpg".format(videoName.split('/')[-1].split('.')[0], str(frameNum))), img)

    return img, subtitle_height_list, canny2_img_list


if __name__ == '__main__':
    test_get_overlap_coordinate()
    pass
