# coding:utf-8
import os
import shutil
import sys
import cv2
import re
import numpy as np

sys.path.append('ctpn')

from math import sqrt, degrees, atan2, fabs, sin, cos, radians
from PIL import Image
from ctpn.text_detect import text_detect
from crnn.crnn import crnnOcr
from crnn_preprocessing import preprocessing
from utils import Queue, get_img_difference, get_union_coordinate


# global constant
# 文本框最小高度（0.045*720=32.4px）
G_MIN_BOX_HEIGHT = 0.045
# 文本框最大高度（0.1*720=72px）
G_MAX_BOX_HEIGHT = 0.1
# 文本最小高度（与preprocessing输出对比）（0.028*720=20.16px）
G_MIN_SUBTITLE_HEIGHT = 0.028
# 文本最大高度（与preprocessing输出对比）（0.070*720=50.4px）
G_MAX_SUBTITLE_HEIGHT = 0.070
# 需要计算限制参数的帧数   # TODO:finetune
G_RAPID_UPDATE_FRAME = 500
# 信息存储在队列中的帧数
G_COMPATE_FRAME_NUM = 2
# 图片相似度阈值   # TODO:finetune
G_DIFFERENT_THRESHOLD = 10

# global variables
# 字幕框高度（每一帧进行计算，初始值：0.04*720=28.8） TODO:finetune
g_subtitle_height = 0.04
g_top = 0.7
g_bottom = 1
g_frame_num_with_subtitle = 0

g_results_list = []
g_canny_img_queue = Queue(G_COMPATE_FRAME_NUM)
g_recs_queue = Queue(G_COMPATE_FRAME_NUM)
# ui显示用的字符串
g_str_ui = ""


def crnnRec(im, text_recs):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box

    """
    results = {}
    xDim, yDim = im.shape[1], im.shape[0]

    for index, rec in enumerate(text_recs):
        results[index] = [rec, ]

        pt1 = (max(1, rec[0]), max(1, rec[1]))
        pt2 = (rec[2], rec[3])
        pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
        pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  ##图像倾斜角度

        partImg = dumpRotateImage(im, degree, pt1, pt2, pt3, pt4)

        image = Image.fromarray(partImg).convert('L')

        sim_pred = crnnOcr(image)

        results[index].append(sim_pred)  ##识别文字

    return results


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), degree, 1)
    matRotation[0, 2] += (widthNew - width) * 0.5
    matRotation[1, 2] += (heightNew - height) * 0.5
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])):min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])):min(xdim - 1, int(pt3[0]))]
    return imgOut


def sort_box(box):
    """
    对box排序,及页面进行排版
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
    """
    box = sorted(box, key=lambda x: sum([x[1]]))  # 按左上角y轴坐标从上到下排序
    return box


# 字幕过滤1
def subtitle_filter1(boxes, img_width, img_height):
    """
    对以下情况的字幕框进行过滤：1.文字位置不在ROI中，2.CTPN检测得到的字幕框高度较小，3.字幕框高度与长度的比值
    :param boxes:
    :param img_height:
    :param img_width:
    :return:
    """

    # roi限制(纵向限制由CTPN完成，即传入CTPN的图像已经过纵向裁剪)
    ROI_X = 0.5
    # 最大字幕框高度与长度比（height/length）TODO:finetune
    MAX_HL_RATIO = 1

    temp = []
    for index, box in enumerate(boxes):
        # 将不满足限制条件的文字框加入到待删除列表中
        if box[0] > img_width * ROI_X:  # 字幕框最左端出现在屏幕右半边
            temp.append(index)
        if (box[5] - box[1]) < img_height * G_MIN_BOX_HEIGHT or (box[5] - box[1]) > img_height * G_MAX_BOX_HEIGHT:  # 字幕框高度较小或较大
            temp.append(index)
        if (box[5] - box[1]) > (box[2] - box[0]) * MAX_HL_RATIO:  # 字幕框高度与长度比值大于1
            temp.append(index)
        # TODO: 除了滚动字幕外出现在中间的字幕框也去掉(离散框)

    boxes = np.delete(boxes, temp, axis=0)
    return sort_box(boxes)


# 获取原始尺寸下的坐标
def convert_to_origin_coordinate(text_recs, resize_ratio):
    tmp = np.zeros((len(text_recs), 8), np.int)
    for index1, text_rec in enumerate(text_recs):
        for index2, point in enumerate(text_rec):
            tmp[index1, index2] = point / resize_ratio
    return tmp


def crop_img(img, video_name, output_path, boxes, frameNum):
    base_name = video_name.split('/')[-1]

    # 对每一帧分别创建一个文件夹存放截取出的字幕
    if os.path.exists(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum))):
        shutil.rmtree(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum)))
    os.makedirs(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum)))

    for i, box in enumerate(boxes):
        left = max(min(box[0], box[4]), 0)
        top = min(box[1], box[3])
        right = min(max(box[2], box[6]), img.shape[1])
        bottom = max(box[5], box[7])
        cropped = img[int(top):int(bottom), int(left):int(right)]  # 高度、宽度
        cv2.imwrite(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                                 "{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, str(i))), cropped)


def delete_overlap_get_scroll_list(text_recs, resize_im_height, output_process=False):
    """
    返回是否为滚动字幕的列表，以及合并了滚动字幕中重叠部分的字幕框
    :param text_recs:
    :param resize_im_height:
    :return:
    """
    global g_str_ui

    no_overlap = []
    uncertain_scroll = []
    is_scroll = []
    result_text_recs = []

    MIN_SCROLL_Y_TOP = 0.88

    for i in range(len(text_recs)):
        # TODO:考虑上方符合要求，但是下方不符合要求的情况，这样处理会出现bug
        # TODO:暂时去除对滚动字幕下方的限制，避免bug
        # if min(text_recs[i][5], text_recs[i][7]) > resize_im_height * 0.93 and min(text_recs[i][1], text_recs[i][3]) > resize_im_height * 0.88:
        if min(text_recs[i][1], text_recs[i][3]) > resize_im_height * MIN_SCROLL_Y_TOP:
                uncertain_scroll.append(i)

    # TODO:考虑上方符合要求，但是下方不符合要求的情况，这样处理会出现bug
    len_of_no_scroll = len(text_recs) - len(uncertain_scroll)
    is_scroll.extend([False] * len_of_no_scroll)
    result_text_recs.extend(text_recs[:len_of_no_scroll])

    if len(uncertain_scroll) > 1:
        no_overlap.append(text_recs[uncertain_scroll[0]])

        for index in uncertain_scroll[1:]:
            for i, rec in enumerate(no_overlap):
                # 检测是否有重叠字幕，有则合并重叠字幕，否则将直接将其添加到no_overlap中
                union = get_union_coordinate(rec, text_recs[index])
                if union:
                    if output_process:
                        g_str_ui += "overlap!!!\tunion_coordinate: " + str(union) + "\n"
                        print("overlap!!!\tunion_coordinate: ", union)
                    # 更新rec到no_scroll
                    no_overlap[i] = [union[2], union[0], union[3], union[0], union[2], union[1], union[3], union[1]]
                    break
            if not union:
                no_overlap.append(text_recs[index])

    elif len(uncertain_scroll) == 1:
        no_overlap.append(text_recs[uncertain_scroll[0]])

    is_scroll.extend([True] * len(no_overlap))
    result_text_recs.extend(no_overlap)
    result_text_recs = sorted(result_text_recs, key=lambda x: sum([x[1]]))

    return is_scroll, result_text_recs


def voting(origin_recs, frame_result, canny_img2_list, origin_img_height, origin_img_width, img_no, output_process=False):
    """
    投票系统，输出相同字幕中出现次数最多的检测结果
    :param origin_recs:
    :param frame_result:
    :param canny_img2_list:
    :param origin_img_height:
    :param origin_img_width:
    :param img_no:
    :param output_process:
    :return:
    """
    global g_str_ui

    # 将当前帧的canny2图像,origin_recs写入队列
    if len(canny_img2_list) > 0:
        if g_canny_img_queue.is_full():  # 如果某个队列已满，需要一起出队一个元素
            g_canny_img_queue.dequeue()
            g_recs_queue.dequeue()
        g_canny_img_queue.enqueue(canny_img2_list)
        g_recs_queue.enqueue(origin_recs)

    # g_results_list:【result_dict_1,result_dict_2...】, result_dict:{result_1:times_1,result_2:times_2,...}
    global g_results_list
    new_results_list = []

    DISTANCE_RESTRICT_PER = 0.01

    max_distance = sqrt(origin_img_height ** 2 + origin_img_width ** 2)
    is_match = False

    if g_canny_img_queue.is_full() and g_results_list:  # 队列满，则说明队列中存有前后两帧的canny_list，可进行相似对比
        for i, curr_canny2 in enumerate(g_canny_img_queue.queue[1]):  # 遍历当前帧的canny_list
            curr_rec = g_recs_queue.queue[1][i]
            # 简化坐标：8 itmes -> 4 items
            curr_top = min(curr_rec[1], curr_rec[3])
            curr_bottom = max(curr_rec[5], curr_rec[7])
            curr_left = min(curr_rec[0], curr_rec[4])
            curr_right = max(curr_rec[2], curr_rec[6])
            simp_curr_rec = [curr_top, curr_bottom, curr_left, curr_right]
            # 计算文本框中心点坐标
            curr_center_x = (curr_left + curr_right) * 0.5
            curr_center_y = (curr_top + curr_bottom) * 0.5
            for j, last_canny2 in enumerate(g_canny_img_queue.queue[0]):  # 遍历前一帧的canny_list
                last_rec = g_recs_queue.queue[0][j]
                # 简化坐标：8 itmes -> 4 items
                last_top = min(last_rec[1], last_rec[3])
                last_bottom = max(last_rec[5], last_rec[7])
                last_left = min(last_rec[0], last_rec[4])
                last_right = max(last_rec[2], last_rec[6])
                simp_last_rec = [last_top, last_bottom, last_left, last_right]
                # 计算文本框中心点
                last_center_x = (last_left + last_right) * 0.5
                last_center_y = (last_top + last_bottom) * 0.5

                # 计算前后两帧中两个文本框中心点的距离
                distance = sqrt((last_center_x - curr_center_x) ** 2 + (last_center_y - curr_center_y) ** 2)

                if distance < DISTANCE_RESTRICT_PER * max_distance:  # 距离小于阈值，能匹配到
                    # 取x轴上的重叠区域
                    x_list = sorted([last_left, last_right, curr_left, curr_right])
                    # 检测前后两帧字幕图像是否重叠
                    if (x_list[0] != x_list[1] and x_list[0] in simp_curr_rec and x_list[1] in simp_curr_rec) or (x_list[0] != x_list[1] and x_list[0] in simp_last_rec and x_list[1] in simp_last_rec):
                        raise ValueError("Two subtitles are not overlapped! One is {}, another is {}".format(str(simp_last_rec), str(simp_curr_rec)))
                    canny_part_xmin = x_list[1]
                    canny_part_xmax = x_list[2]
                    canny_len = canny_part_xmax - canny_part_xmin

                    # 取y轴上的重叠区域
                    y_list = sorted([last_top, last_bottom, curr_top, curr_bottom])
                    # 检测前后两帧字幕图像是否重叠
                    if (y_list[0] != y_list[1] and y_list[0] in simp_curr_rec and y_list[1] in simp_curr_rec) or (y_list[0] != y_list[1] and y_list[0] in simp_last_rec and y_list[1] in simp_last_rec):
                        raise ValueError("Two subtitles are not overlapped! One is {}, another is {}".format(str(simp_last_rec), str(simp_curr_rec)))
                    canny_part_ymin = y_list[1]
                    canny_part_ymax = y_list[2]

                    # 在x轴上缩小对比图像的面积
                    compute_part_len = canny_len / 2 // 2
                    x_min = (canny_part_xmin + canny_part_xmax) // 2 - compute_part_len
                    x_max = (canny_part_xmin + canny_part_xmax) // 2 + compute_part_len

                    # 转换成相对坐标
                    related_last_xmin = int(x_min - last_left)
                    related_last_xmax = int(x_max - last_left)
                    related_curr_xmin = int(x_min - curr_left)
                    related_curr_xmax = int(x_max - curr_left)
                    related_last_ymin = int(canny_part_ymin - last_top)
                    related_last_ymax = int(canny_part_ymax - last_top)
                    related_curr_ymin = int(canny_part_ymin - curr_top)
                    related_curr_ymax = int(canny_part_ymax - curr_top)

                    # 计算图片相似度
                    old_img = g_canny_img_queue.queue[0][j][related_last_ymin:related_last_ymax,
                                                            related_last_xmin:related_last_xmax]
                    new_img = g_canny_img_queue.queue[1][i][related_curr_ymin:related_curr_ymax,
                                                            related_curr_xmin:related_curr_xmax]
                    difference = get_img_difference(old_img, new_img, hash_type="perception")

                    if output_process:
                        g_str_ui += "the difference between" + str(img_no) + "_" + str(i) + "and " + str(img_no - 1) + "_" + str(j) + ":" + str(difference) + "\n"
                        print("the difference between", str(img_no), "_", str(i), "and ", str(img_no - 1), "_", str(j),
                              ":", str(difference))

                    if difference >= G_DIFFERENT_THRESHOLD:  # 匹配到，但是计算相似度的结果说明两个字幕不同
                        # from random import uniform, randint
                        # r = round(uniform(0, 1), 3)
                        # cv2.imwrite(os.path.join(output_path, "{}_{}_part_{}_{}.jpg".format(img_no, img_no - 1, r, round(uniform(0, 1), 3))), old_img)
                        # cv2.imwrite(os.path.join(output_path, "{}_{}_part_{}_{}.jpg".format(img_no, img_no - 1, r, round(uniform(0, 1), 3))), new_img)
                        if output_process:
                            g_str_ui += str(img_no) + "_" + str(i) + " different from" + str(img_no - 1) + "_" + str(j) + "\n"
                            print(str(img_no), "_", str(i), " different from", str(img_no - 1), "_", str(j))
                        # 去掉上一帧同位置的投票结果（不从原来的结果中读取前一帧的投票结果），并新增当前帧结果
                        new_results_list.append({frame_result[i][1]: 1})

                    else:  # 匹配到，计算相似度的结果说明两个字幕相同
                        # 更新位置（按y轴坐标进行排序写入list），结果（value）+1[{result1:times1,result2:times2,...},{result1:times1,result2:times2,...},...]
                        result_dict = g_results_list[j]  # 读取该文本框在前一帧中的投票结果
                        if frame_result[i][1] in result_dict:  # 该结果之前已存在，数值+1
                            result_dict[frame_result[i][1]] = int(result_dict[frame_result[i][1]]) + 1
                        else:  # 该结果之前不存在，新增一个投票项，数值为1
                            result_dict[frame_result[i][1]] = 1
                        new_results_list.append(result_dict)
                    is_match = True
                    break
                else:  # 距离大于阈值，没匹配到
                    is_match = False
            # j遍历后都未被匹配，新建一个结果
            if not is_match:
                result_dict = {frame_result[i][1]: 1}
                new_results_list.append(result_dict)
                is_match = True
        # 用当前帧结果替代前一帧结果以去除结果中未被比较的项，只保留本帧中出现字幕位置的结果
        g_results_list = new_results_list
    else:  # 队列未满，则说明这是第一帧，需要初始化RESULT_LIST
        for rlt in frame_result:
            g_results_list.append({rlt[1]: 1})
    if output_process:
        g_str_ui += str(g_results_list) + "\n"
        print(g_results_list)


def model_news(img, img_no, video_name, output_path, output_process=False):
    global g_str_ui
    g_str_ui = ''

    origin_img = img.copy()
    origin_img_height = origin_img.shape[0]
    origin_img_width = origin_img.shape[1]

    # ctpn
    text_recs, drawn_img, resize_img, resize_ratio = text_detect(origin_img, top=0.5, bottom=1, left=0, right=1)

    resize_im_height = resize_img.shape[0]
    resize_im_width = resize_img.shape[1]

    # 第一次字幕过滤(位置信息)
    text_recs = subtitle_filter1(text_recs, resize_im_width, resize_im_height)
    if text_recs is None or len(text_recs) == 0:
        return [], origin_img, [], resize_ratio, g_str_ui

    # 合并滚动字幕中重叠部分，并获取滚动字幕标记
    is_scroll, text_recs = delete_overlap_get_scroll_list(text_recs, resize_im_height, output_process=output_process)
    # TODO:画出重叠区域的框，验证算法是否正确

    origin_recs = convert_to_origin_coordinate(text_recs, resize_ratio)

    if output_process:
        crop_img(img, video_name, output_path, origin_recs, img_no)

    # crnn前预处理
    origin_img = img.copy()
    preprocessed_img, canny_img2_list = preprocessing.p_picture(origin_recs, is_scroll, origin_img, img_no, video_name, output_path)

    # 送入CRNN检测
    result = crnnRec(resize_img, text_recs)
    # 去除检测结果最前端非中文字符,出现重复字符的彻底解决方法应为重新训练
    for key in result:
        pattern = re.compile(u"^[^\u4e00-\u9fa5]+")
        result[key][1] = re.sub(pattern, '', result[key][1])

    # 根据位置对比是否同一字幕，进行投票
    # 去除is_scroll为True的字幕框，不进行投票
    no_scroll_result = []
    no_scroll_canny_list = []
    no_scroll_recs = []

    for i in result:
        if not is_scroll[i]:
            no_scroll_result.append(result[i])
            no_scroll_canny_list.append(canny_img2_list[i])
            no_scroll_recs.append(origin_recs[i])

    # 投票
    voting(no_scroll_recs, no_scroll_result, no_scroll_canny_list, origin_img_height, origin_img_width, img_no, output_process=output_process)

    # 输出出现次数最多的结果
    for index, data in enumerate(g_results_list):
        for k in sorted(data, key=data.__getitem__, reverse=True):
            if output_process:
                g_str_ui += "result:" + k + ", times:" + str(data[k]) + "\n"
                print("result:" + k + ", times:" + str(data[k]))
            result[int(index)][1] = k
            break

    if output_process:
        return result, preprocessed_img, is_scroll, resize_ratio, g_str_ui
    else:
        return result, origin_img, is_scroll, resize_ratio, g_str_ui
