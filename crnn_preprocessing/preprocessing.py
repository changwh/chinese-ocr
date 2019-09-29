# coding=utf-8
import numpy as np
import cv2 as cv
import sys
import os
import random
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


# def erode(img, erode_kernel):
#     ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (erode_kernel, erode_kernel))      # kernel(5,5)
#     dst = cv.erode(thresh, kernel)
#     return dst
#
#
# def get_font_weight_from_random_point(canny_img2, white_point):
#     ss = random.randint(0, len(white_point) - 1)
#     row, col = white_point[ss]
#     if canny_img2[row, col] != 0:
#         return 0
#     else:
#         max_line = canny_img2.shape[0]
#         max_col = canny_img2.shape[1]
#
#         top, bottom, left, right = 0, 0, 0, 0
#         if canny_img2[row, col] == 0:
#
#             for i in range(row - 1, 0 - 1, -1):
#                 if canny_img2[i, col] == 0:
#                     top += 1
#                 else:
#                     break
#
#             for i in range(row + 1, max_line):
#                 if canny_img2[i, col] == 0:
#                     bottom += 1
#                 else:
#                     break
#
#             for j in range(col - 1, 0 - 1, -1):
#                 if canny_img2[row, j] == 0:
#                     left += 1
#                 else:
#                     break
#
#             for j in range(col + 1, max_col):
#                 if canny_img2[row, j] == 0:
#                     right += 1
#                 else:
#                     break
#
#         vertical = top + bottom
#         horizontal = left + right
#         font_weight = vertical if vertical < horizontal else horizontal
#
#         if 1 < font_weight < 10:
#             return font_weight + 1
#         else:
#             return 0
#
#
# def get_the_font_weight(gray_image, canny_img2):
#     _, thresh1 = cv.threshold(gray_image, 245, 255, cv.THRESH_BINARY)
#
#     row = thresh1.shape[0]
#     col = thresh1.shape[1]
#     white_point = []
#     for i in range(row):
#         for j in range(col):
#             if thresh1[i, j] == 255:
#                 white_point.append((i, j))
#
#     if len(white_point) > 0:
#         font_weight_list = np.arange(50)
#         char_point_num = 0
#         for i in range(300):
#             if char_point_num >= 50:
#                 break
#             font_weight = get_font_weight_from_random_point(canny_img2, white_point)
#             if font_weight != 0:
#                 font_weight_list[char_point_num] = font_weight
#                 char_point_num += 1
#
#         counts = np.bincount(font_weight_list)
#         final_font_weight = np.argmax(counts)
#         return final_font_weight
#     else:
#         return 0
#
#
# def get_converted_result(step2, binary_img):
#     row = binary_img.shape[0]
#     col = binary_img.shape[1]
#
#     for i in range(row):
#         for j in range(col):
#             if step2[i, j] == 0:
#                 binary_img[i, j] = [0, 0, 0]
#             binary_img[i, j] = 255 - binary_img[i, j]
#     return binary_img
#
#
# def precess_2(img):
#     row = img.shape[0]
#     col = img.shape[1]
#     i = 0
#     j = 0
#     while (j < col):
#         if img[0, j, 0] == 255:
#             # global line_extent
#             # line_extent = 0
#             clean_the_line(img, 1, j, row, col)
#             # print(line_extent)
#         j = j + 1
#
#     i = 0
#     j = 0
#     while (j < col):
#         if img[row - 1, j, 0] == 255:
#             # global line_extent
#             # line_extent = 0
#             clean_the_line(img, row - 2, j, row, col)
#             # print(line_extent)
#         j = j + 1
#     i = 0
#     j = 0
#     while (i < row):
#         if img[i, j, 0] == 255:
#             # global line_extent
#             # line_extent = 0
#             clean_the_line(img, i, j, row, col)
#             # print(line_extent)
#         i = i + 1
#
#     return img
#
#
# def clean_the_line(img, i, j, row, col):
#     img[i, j] = [0, 0, 0]
#     # global line_extent
#     # line_extent = line_extent + 1
#     # if line_extent>100:
#     #   return
#     # print(i, j)
#     if img[i, j + 1, 0] == 255 and j != col - 2:  # 右边
#         clean_the_line(img, i, j + 1, row, col)
#     if img[i + 1, j + 1, 0] == 255 and j != col - 2 and i != row - 2:  # 右下
#         clean_the_line(img, i + 1, j + 1, row, col)
#     if img[i + 1, j, 0] == 255 and i != row - 2:  # 下边
#         clean_the_line(img, i + 1, j, row, col)
#     if img[i + 1, j - 1, 0] == 255 and i != row - 2 and j != 1:  # 左下
#         clean_the_line(img, i + 1, j - 1, row, col)
#     if img[i, j - 1, 0] == 255 and j != 1:  # 左边
#         clean_the_line(img, i, j - 1, row, col)
#     if img[i - 1, j + 1, 0] == 255 and j != col - 2 and i != 1:  # 右上
#         clean_the_line(img, i - 1, j + 1, row, col)
#     if img[i - 1, j, 0] == 255 and i != 1:  # 上边
#         clean_the_line(img, i - 1, j, row, col)
#     if img[i - 1, j - 1, 0] == 255 and i != 1 and j != 1:  # 左上
#         clean_the_line(img, i - 1, j - 1, row, col)
#     return 0
#
#
# def convert(image):
#     height = image.shape[0]
#     width = image.shape[1]
#
#     for i in range(height):
#         for j in range(width):
#             image[i, j] = 255 - image[i, j]
#     return image
#
#
# def get_the_font_weight_v2(gray_image, canny_img2):
#     _, thresh1 = cv.threshold(gray_image, 245, 255, cv.THRESH_BINARY)
#
#     row = thresh1.shape[0]
#     col = thresh1.shape[1]
#     white_point = []
#     for i in range(row):
#         for j in range(col):
#             if thresh1[i, j] == 255:
#                 white_point.append((i, j))
#
#     if len(white_point) > 0:
#         font_weight_list = np.arange(50)
#         char_point_num = 0
#         for i in range(300):
#             if char_point_num >= 50:
#                 break
#             font_weight = get_font_weight_from_random_point(canny_img2, white_point)
#             if font_weight != 0:
#                 font_weight_list[char_point_num] = font_weight
#                 char_point_num += 1
#
#         counts = np.bincount(font_weight_list)
#         final_font_weight = np.argmax(counts)
#         # TODO：？？？
#         y = 0
#         for x in font_weight_list:
#             y = x+y
#         if y>500:
#             return 30
#         return final_font_weight
#     else:
#         return 0


def main(left, top, right, bottom, img, videoName, outputPath, frameNum, index):
    base_name = videoName.split('/')[-1]

    roi = img[top:bottom, left:right]

    # 读取灰度图
    gray_image = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # 边缘检测得到边缘图片
    canny_img = cv.Canny(roi, 40, 120)  # 与视频清晰度相关,清晰度越高,阈值可相应调高(1:3)
    # cv.imshow('canny 1', canny_img)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "1_canny_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img)

    # 阈值处理
    _, thresh1 = cv.threshold(gray_image, 225, 255, cv.THRESH_BINARY)
    # cv.imshow('thresh1!', thresh1)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "2_gray_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), thresh1)

    # 膨胀
    dilate_img = dilate(thresh1)
    # cv.imshow('dilate_demo', thresh2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "3_dilate_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), dilate_img)

    # 对canny1图像使用阈值处理膨胀后的图像过滤
    canny_img2 = filter_canny_with_thresh(dilate_img, canny_img)

    # # 获取笔画宽度，为选取膨胀、腐蚀参数做准备
    # font_weight = get_the_font_weight(gray_image, canny_img2)

    # 去除上下（左右暂时不用）两边的canny噪声
    canny_img2, subtitle_height = wash_canny_picture(canny_img2)

    # 递归去除边缘噪声？
    # canny_img2 = precess_2(canny_img2)
    # cv.imshow('canny 2', canny_img2)

    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)

    # # cv.imwrite(os.path.join(outputPath, "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)
    # # np.savetxt(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
    # #                         "4_canny2_{}_{}_{}.txt".format(base_name.split('.')[0], frameNum, index)), canny_img2, fmt='%d')
    #
    # # TODO:需要通过测试获得更多的参数
    # if font_weight == 3:
    #     dilate_kernel = 5
    #     erode_kernel = 3
    # elif font_weight == 4:
    #     dilate_kernel = 7
    #     erode_kernel = 4
    # elif font_weight == 5:
    #     dilate_kernel = 8
    #     erode_kernel = 5
    # elif font_weight <= 2:
    #     dilate_kernel = 4
    #     erode_kernel = 2
    # elif font_weight == 6:
    #     dilate_kernel = 9
    #     erode_kernel = 6
    # elif font_weight >= 7:
    #     dilate_kernel = 10
    #     erode_kernel = 7
    #
    # dilate_img2 = dilate(canny_img2, dilate_kernel=dilate_kernel)
    # erode_img = erode(dilate_img2, erode_kernel)
    #
    # cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
    #                         "5_erode_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), erode_img)
    #
    # final = get_converted_result(erode_img, roi)
    #
    # # img[top:bottom, left:right] = final
    #
    # # cv.imshow('final!!!!!', img)
    # # if cv.waitKey(0) == ord('q'):
    # #     pass
    # cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
    #                         "6_output_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), final)
    # # cv.imwrite(os.path.join(outputPath, "final_{}_{}.jpg".format(base_name.split('.')[0], str(frameNum))), img)
    # return final, subtitle_height, canny_img2

    return subtitle_height, canny_img2

def main_v2(left, top, right, bottom, img, videoName, outputPath, frameNum, index):
    base_name = videoName.split('/')[-1]

    roi = img[top:bottom, left:right]
    # roi = copy.deepcopy(roi_real)

    #读取灰度图
    gray_image = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    # gray_image = copy.deepcopy(gray_image_real)

    # 边缘检测得到边缘图片
    canny_img = cv.Canny(gray_image, 40, 120)  # 40 120
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "1_canny_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img)

    # #TODO:分离滚动字幕前的固定字幕
    # #是否需要分离
    # # Is_roll = True
    # Is_roll = False
    # if (Is_roll == True):
    #     img2_0, line_num = news_procress(canny_img, roi)
    #     # left = left+line_num
    #     return 0

    check_out_value, thresh_cc = conculate_proportion_v3(roi)  # 0.4
    # picture_name = str(base_name)+'_'+str(frameNum)+'_'+str(index)

    # 阈值处理
    if 0 == check_out_value:    # 简单背景（不需要处理,canny1作为canny2输出用于图片相似度对比）
        # print(picture_name + ' is with background color!')
        # im_at_mean = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 10)
        # return roi, -1, canny_img
        return canny_img
    elif 1 == check_out_value:    # 复杂背景彩色字（使用另外的方法获得阈值处理后的图像）
        # print(picture_name + ' is without background color!And it is not white')
        thresh1 = thresh_cc
        # gray_image = erode(thresh_cc, 3)
    elif 2 == check_out_value:    # 复杂背景白色字（使用V1的方法获得阈值处理后的图像）
        # print(picture_name +' is without background color! And it is white')
        _, thresh1 = cv.threshold(gray_image, 200, 255, cv.THRESH_BINARY)

    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "2_thresh1_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), thresh1)

    #膨胀
    dilate_img = dilate(thresh1)
    # cv.imshow('dilate_demo', thresh2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "3_dilate_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), dilate_img)

    # 对canny1图像使用阈值处理膨胀后的图像过滤
    canny_img2 = filter_canny_with_thresh(dilate_img, canny_img)  #0.2s

    # if 1 == check_out_value:
    #     gray_image = erode(thresh_cc, 3)

    # # 获取笔画宽度，为选取膨胀、腐蚀参数做准备
    # font_weight = get_the_font_weight(gray_image, canny_img2) #0.25s

    # 去除上下（左右暂时不用）两边的canny噪声
    canny_img2, subtitle_height = wash_canny_picture(canny_img2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)

    # # 先判断是否正确，the_width=30说明预知检测出现了问题，需要反色
    # if font_weight == 30:
    #     convert(gray_image)
    #     # cv.imshow('ss',gray_image)
    #     # the_width = get_the_width(gray_image, candy_img2)
    #
    # # print('The width is: ' + str(the_width))
    #
    # # TODO:需要通过测试获得更多的参数
    # if font_weight == 3:
    #     dilate_kernel = 5
    #     erode_kernel = 3
    # elif font_weight == 4:
    #     dilate_kernel = 7
    #     erode_kernel = 4
    # elif font_weight == 5:
    #     dilate_kernel = 8
    #     erode_kernel = 5
    # elif font_weight <= 2:
    #     dilate_kernel = 4
    #     erode_kernel = 2
    # elif font_weight == 6:
    #     dilate_kernel = 9
    #     erode_kernel = 6
    # elif font_weight >= 7:
    #     dilate_kernel = 10
    #     erode_kernel = 7
    #
    # dilate_img2 = dilate(canny_img2, dilate_kernel=dilate_kernel)
    # erode_img = erode(dilate_img2, erode_kernel)
    # cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
    #                         "5_erode_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), erode_img)
    #
    # final = get_converted_result(erode_img, roi)
    #
    # cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
    #                         "6_output_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), final)

    # return final, subtitle_height, canny_img2
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


def p_picture(origin_recs, is_scroll, origin_img, frameNum, videoName, outputPath, version):
    subtitle_height_list = []
    canny2_img_list = []
    output_list = []

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
        # TODO:拆分
        if version == 1:
            subtitle_height, canny2_img = main(left, top, right, bottom, origin_img, videoName, outputPath, frameNum, index)
            subtitle_height_list.append(subtitle_height)
        elif version == 2:  # TODO:remove subtitle_height, output
            # output, subtitle_height, canny2_img = main_v2(left, top, right, bottom, origin_img, videoName, outputPath, frameNum, index)
            # output_list.append(output)
            canny2_img = main_v2(left, top, right, bottom, origin_img, videoName, outputPath, frameNum, index)

        canny2_img_list.append(canny2_img)

    # # 处理重叠部分
    # overlap_part = []
    # overlap_coordinate_list = []
    #
    # for index in range(len(origin_recs) - 1):
    #     # 计算当前文本框与下一文本框是否存在重叠部分，若存在，获取重叠部分坐标
    #     overlap_coordinate = get_overlap_coordinate(origin_recs[index], origin_recs[index + 1])
    #     if overlap_coordinate:
    #         # 计算重叠部分在当前文本框中的相对位置
    #         relative_top = overlap_coordinate[0] - origin_recs[index][1]
    #         relative_bottom = overlap_coordinate[1] - origin_recs[index][1]
    #         relative_left = overlap_coordinate[2] - origin_recs[index][0]
    #         relative_right = overlap_coordinate[3] - origin_recs[index][0]
    #         overlap_part_1 = output_list[index][relative_top:relative_bottom, relative_left:relative_right]
    #         overlap_p1_copy = overlap_part_1.copy()
    #
    #         # 计算重叠部分在下一文本框中的相对位置
    #         relative_top = overlap_coordinate[0] - origin_recs[index + 1][1]
    #         relative_bottom = overlap_coordinate[1] - origin_recs[index + 1][1]
    #         relative_left = overlap_coordinate[2] - origin_recs[index + 1][0]
    #         relative_right = overlap_coordinate[3] - origin_recs[index + 1][0]
    #         overlap_part_2 = output_list[index + 1][relative_top:relative_bottom, relative_left:relative_right]
    #         overlap_p2_copy = overlap_part_2.copy()
    #
    #         # 将两个文本框的重叠部分中的非白色部分进行整合
    #         height = relative_bottom - relative_top
    #         width = relative_right - relative_left
    #         for i in range(height):
    #             for j in range(width):
    #                 if overlap_p2_copy[i, j].any() != 255:
    #                     overlap_p1_copy[i, j] = overlap_p2_copy[i, j]
    #
    #         overlap_coordinate_list.append(overlap_coordinate)
    #         overlap_part.append(overlap_p1_copy)
    #
    #     # 将处理结果拼回原图（不考虑重叠问题）
    #     origin_img[location_list[index][0]:location_list[index][1], location_list[index][2]:location_list[index][3]] = output_list[index]
    # origin_img[location_list[-1][0]:location_list[-1][1], location_list[-1][2]:location_list[-1][3]] = output_list[-1]
    # # 将整合后的重叠部分拼回原图
    # for index in range(len(overlap_part)):
    #     origin_img[overlap_coordinate_list[index][0]:overlap_coordinate_list[index][1], overlap_coordinate_list[index][2]:overlap_coordinate_list[index][3]] = overlap_part[index]

    drawn_origin_img = draw_ctpn_result_boxes(location_list, origin_img)
    cv.imwrite(os.path.join(outputPath, "final_{}_{}.jpg".format(videoName.split('/')[-1].split('.')[0], str(frameNum))), drawn_origin_img)

    if version == 1:
        return drawn_origin_img, subtitle_height_list, canny2_img_list
    else:
        return drawn_origin_img, canny2_img_list


# if __name__ == '__main__':
#     # test_get_overlap_coordinate()
#     pass
