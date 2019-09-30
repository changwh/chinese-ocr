# coding=utf-8
import numpy as np
import cv2 as cv
import sys
import os
from .clean_image import wash_canny_picture
from .choose_color import conculate_proportion_v3
from utils import draw_ctpn_result_boxes

np.set_printoptions(threshold=sys.maxsize)


def dilate(img, dilate_kernel=5):
    _, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_kernel, dilate_kernel))  # kernel(5, 5)
    dst = cv.dilate(thresh, kernel)
    return dst


def filter_canny_with_thresh(thresh1, canny_img):
    row = thresh1.shape[0]
    col = thresh1.shape[1]

    for i in range(row):
        for j in range(col):
            if thresh1[i, j] == 255 and canny_img[i, j] == 255:
                canny_img[i, j] = 255
            else:
                canny_img[i, j] = 0
    return canny_img


def main(left, top, right, bottom, img, videoName, outputPath, frameNum, index):
    base_name = videoName.split('/')[-1]

    roi = img[top:bottom, left:right]

    #读取灰度图
    gray_image = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # 边缘检测得到边缘图片
    canny_img = cv.Canny(gray_image, 40, 120)  # 40 120
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "1_canny_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img)

    check_out_value, thresh_cc = conculate_proportion_v3(roi)  # 0.4

    # 阈值处理
    if 0 == check_out_value:    # 简单背景（不需要处理,canny1作为canny2输出用于图片相似度对比）
        return canny_img
    elif 1 == check_out_value:    # 复杂背景彩色字（使用另外的方法获得阈值处理后的图像）
        thresh1 = thresh_cc
    elif 2 == check_out_value:    # 复杂背景白色字（使用V1的方法获得阈值处理后的图像）
        _, thresh1 = cv.threshold(gray_image, 200, 255, cv.THRESH_BINARY)

    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "2_thresh1_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), thresh1)

    # 膨胀
    dilate_img = dilate(thresh1)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "3_dilate_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), dilate_img)

    # 对canny1图像使用阈值处理膨胀后的图像过滤
    canny_img2 = filter_canny_with_thresh(dilate_img, canny_img)  #0.2s

    # 去除上下（左右暂时不用）两边的canny噪声
    canny_img2, subtitle_height = wash_canny_picture(canny_img2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)

    return canny_img2


def get_localtions(text_recs, img):
    locations = []
    for index in range(len(text_recs)):
        left = max(min(text_recs[index, 0], text_recs[index, 4]), 0)
        top = min(text_recs[index, 1], text_recs[index, 3])
        right = min(max(text_recs[index, 2], text_recs[index, 6]), img.shape[1])
        bottom = max(text_recs[index, 5], text_recs[index, 7])
        locations.append([top, bottom, left, right])
    return locations


def p_picture(origin_recs, is_scroll, origin_img, frameNum, videoName, outputPath):
    canny2_img_list = []

    # 提取CTPN检测结果中的坐标
    location_list = get_localtions(origin_recs, origin_img)
    # 进行预处理
    for index in range(len(origin_recs)):
        if is_scroll[index]:
            continue
        top = location_list[index][0]
        bottom = location_list[index][1]
        left = location_list[index][2]
        right = location_list[index][3]

        canny2_img = main(left, top, right, bottom, origin_img, videoName, outputPath, frameNum, index)

        canny2_img_list.append(canny2_img)

    drawn_origin_img = draw_ctpn_result_boxes(location_list, origin_img)
    cv.imwrite(os.path.join(outputPath, "final_{}_{}.jpg".format(videoName.split('/')[-1].split('.')[0], str(frameNum))), drawn_origin_img)

    return drawn_origin_img, canny2_img_list

