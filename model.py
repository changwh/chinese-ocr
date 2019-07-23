# coding:utf-8
import os
import shutil
import sys
import cv2
import re
import numpy as np

sys.path.append('ctpn')

from math import *
from PIL import Image
from random import randint
from ctpn.text_detect import text_detect
from crnn.crnn import crnnOcr
from crnn_preprocessing import preprocessing
from ctpn.ctpn.cfg import Config
from ctpn.ctpn.other import resize_im
from crnn_preprocessing.img_difference import get_img_difference


class Queue():

    def __init__(self, size):
        self.size = size
        self.front = -1
        self.rear = -1
        self.queue = []

    def enqueue(self, ele):  # 入队操作
        if self.is_full():
            # raise exception("queue is full")
            pass
        else:
            self.queue.append(ele)
            self.rear = self.rear + 1

    def dequeue(self):  # 出队操作
        if self.is_empty():
            # raise exception("queue is empty")
            pass
        else:
            self.queue.pop(0)
            self.front = self.front + 1

    def is_full(self):
        return self.rear - self.front == self.size

    def is_empty(self):
        return self.front == self.rear

    def clear_queue(self):
        self.rear = self.front = -1
        self.queue = []

    def show_queue(self):
        print(self.queue)


# global variables
CANNY_IMG_QUEUE = Queue(2)
HEIGHT_OF_SUBTITLE_FILTER_PER = 0.04
LEFT_TOP_Y_PER = 0.7
LEFT_BOTTOM_Y_PER = 1
COUNT_OF_FRAME_WITH_SUBTITLE = 0
RESULTS_DICT = {}
COUNT_OF_LOOSE_FRAME = 500  # TODO:finetune
DIFFERENT_THRESHOLD = 6  # TODO:finetune


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
    # height,width=imgOut.shape[:2]
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


# 字幕过滤
def subtitle_filter(boxes, img_height, img_width, real_height,
                    subtitle_height_list=None, canny_img2_list=None, seq=0, output_process=False):
    # roi限制
    roi_x_per = 0.5
    # 条件收束前字幕大小限制
    height_min_delta = 0.01
    height_max_delta = 0.03
    # 条件收束后字幕大小限制
    height_delta_strict = 0.01
    # 条件收束后字幕高度限制
    subtitle_location_y_delta = 0.015

    temp = []
    if seq == 0:
        for index, box in enumerate(boxes):
            # 将不满足限制条件的文字框加入到待删除列表中
            if box[0] > img_width * roi_x_per:  # 字幕框最左端出现在屏幕右半边
                temp.append(index)
            if (box[5] - box[1]) / real_height < 0.045:  # 字幕框高度较小 TODO:finetune
                temp.append(index)
            if (box[5] - box[1]) / (box[2] - box[0]) > 1:  # 字幕框高度与长度比值大于1 TODO:finetune
                temp.append(index)
            # TODO: 除了滚动字幕外出现在中间的字幕框也去掉(离散框)

        boxes = np.delete(boxes, temp, axis=0)
        return sort_box(boxes)
    elif seq == 1:
        for index, subtitle_height in enumerate(subtitle_height_list):
            if COUNT_OF_FRAME_WITH_SUBTITLE < COUNT_OF_LOOSE_FRAME:
                if subtitle_height < real_height * (HEIGHT_OF_SUBTITLE_FILTER_PER - height_min_delta) \
                        or subtitle_height > real_height * (HEIGHT_OF_SUBTITLE_FILTER_PER + height_max_delta):
                    if output_process:
                        print("loose restriction")
                        print("subtitle_height:" + str(subtitle_height) + ", restriction: max:" + str(
                            real_height * (HEIGHT_OF_SUBTITLE_FILTER_PER + height_max_delta)) + " min:" + str(
                            real_height * (HEIGHT_OF_SUBTITLE_FILTER_PER - height_min_delta)))
                    temp.append(index)
            else:
                if abs(subtitle_height / real_height - HEIGHT_OF_SUBTITLE_FILTER_PER) > height_delta_strict \
                        or boxes[index][1] < (LEFT_TOP_Y_PER - subtitle_location_y_delta) * img_height \
                        or boxes[index][7] > (LEFT_BOTTOM_Y_PER + subtitle_location_y_delta) * img_height:
                    if output_process:
                        print("tight restriction")
                        print("subtitle_height:" + str(subtitle_height) + ", restriction: max:" + str(
                            real_height * (HEIGHT_OF_SUBTITLE_FILTER_PER + height_delta_strict)) + " , min:" + str(
                            real_height * (HEIGHT_OF_SUBTITLE_FILTER_PER - height_delta_strict)))
                        print("boxes top:" + str(boxes[index][1]) + ", restriction: min:" + str(
                            (LEFT_TOP_Y_PER - subtitle_location_y_delta) * img_height))
                        print("boxes bottom:" + str(boxes[index][7]) + ", restriction: max:" + str(
                            (LEFT_BOTTOM_Y_PER + subtitle_location_y_delta) * img_height))
                    temp.append(index)

        boxes = np.delete(boxes, temp, axis=0)
        canny_img2_list = [canny_img2_list[i] for i in range(len(canny_img2_list)) if (i not in temp)]
        return boxes, canny_img2_list


# 获取原始尺寸下的坐标
def toRealCoordinate(text_recs, f):
    tmp = np.zeros((len(text_recs), 8), np.int)

    for index1, text_rec in enumerate(text_recs):
        for index2, point in enumerate(text_rec):
            tmp[index1, index2] = point / f

    return tmp


def crop_img(img, video_name, output_path, boxes, frameNum):
    base_name = video_name.split('/')[-1]

    # 对每一帧分别创建一个文件夹存放截取出的字幕
    if os.path.exists(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum))):
        shutil.rmtree(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum)))
    os.makedirs(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum)))

    i = 0

    for box in boxes:
        left = max(min(box[0], box[4]), 0)
        top = min(box[1], box[3])
        right = min(max(box[2], box[6]), img.shape[1])
        bottom = max(box[5], box[7])
        cropped = img[int(top):int(bottom), int(left):int(right)]  # 高度、宽度
        cv2.imwrite(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                                 "{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, str(i))), cropped)
        i = i + 1


def model(img, imgNo, videoName, outputPath, output_process=False):
    real_img = img.copy()
    real_img_height = real_img.shape[0]
    real_img_width = real_img.shape[1]

    text_recs, drawn_img, img, f = text_detect(img, top=0.7, bottom=1, left=0, right=1)

    resize_im_height = img.shape[0]
    resize_im_width = img.shape[1]

    # 第一次字幕过滤(位置信息)
    text_recs = subtitle_filter(text_recs, resize_im_height, resize_im_width, real_img_height, seq=0, output_process=output_process)
    if text_recs is None or len(text_recs) == 0:
        return [], real_img, [], f

    # 获取原图坐标便于预处理
    real_recs = toRealCoordinate(text_recs, f)

    if output_process:
        crop_img(real_img, videoName, outputPath, real_recs, imgNo)

    # crnn前预处理
    tmp = real_img.copy()
    preprocessed_img, subtitle_height_list, canny_img2_list = preprocessing.p_picture(real_recs, [], tmp, imgNo,
                                                                                      videoName, outputPath)
    if output_process:
        np.savetxt(
            os.path.join(outputPath, "subtitle_height_{}_{}.txt".format(videoName.split('/')[-1].split('.')[0], imgNo)),
            subtitle_height_list, fmt='%d')

    # 每一帧中只随机抽取一个高度,避免某一帧中非字幕文本对平均高度的影响,通过前500帧含有字幕的图像获取字幕高度
    global HEIGHT_OF_SUBTITLE_FILTER_PER
    global LEFT_TOP_Y_PER
    global LEFT_BOTTOM_Y_PER
    global COUNT_OF_FRAME_WITH_SUBTITLE
    global RESULTS_DICT

    min_height_subtitle = 0.028 * real_img_height  # TODO:finetune

    if subtitle_height_list and COUNT_OF_FRAME_WITH_SUBTITLE < COUNT_OF_LOOSE_FRAME:
        index = randint(0, len(subtitle_height_list) - 1)  # 同一帧中检测到多个疑似字幕,则从中随机选择一个进行数据更新
        if subtitle_height_list[index] > min_height_subtitle:
            HEIGHT_OF_SUBTITLE_FILTER_PER = 0.9 * HEIGHT_OF_SUBTITLE_FILTER_PER + 0.1 * subtitle_height_list[
                index] / real_img_height
            LEFT_TOP_Y_PER = 0.9 * LEFT_TOP_Y_PER + 0.1 * text_recs[index][1] / resize_im_height
            LEFT_BOTTOM_Y_PER = 0.9 * LEFT_BOTTOM_Y_PER + 0.1 * text_recs[index][7] / resize_im_height
            COUNT_OF_FRAME_WITH_SUBTITLE = COUNT_OF_FRAME_WITH_SUBTITLE + 1
    elif subtitle_height_list and COUNT_OF_FRAME_WITH_SUBTITLE >= COUNT_OF_LOOSE_FRAME:
        index = randint(0, len(subtitle_height_list) - 1)  # 同一帧中检测到多个疑似字幕,则从中随机选择一个进行数据更新
        if subtitle_height_list[index] > min_height_subtitle:
            HEIGHT_OF_SUBTITLE_FILTER_PER = 0.95 * HEIGHT_OF_SUBTITLE_FILTER_PER + 0.05 * subtitle_height_list[
                index] / real_img_height
            LEFT_TOP_Y_PER = 0.95 * LEFT_TOP_Y_PER + 0.05 * text_recs[index][1] / resize_im_height
            LEFT_BOTTOM_Y_PER = 0.95 * LEFT_BOTTOM_Y_PER + 0.05 * text_recs[index][7] / resize_im_height
            COUNT_OF_FRAME_WITH_SUBTITLE = COUNT_OF_FRAME_WITH_SUBTITLE + 1

    if output_process:
        print("height of subtitle:" + str(HEIGHT_OF_SUBTITLE_FILTER_PER))
        print("left_top_y:" + str(LEFT_TOP_Y_PER))
        print("left_bottom_y:" + str(LEFT_BOTTOM_Y_PER))

    # 第二次字幕过滤(高度,更精确的纵坐标)
    text_recs, canny_img2_list = subtitle_filter(text_recs, resize_im_height, resize_im_width, real_img_height,
                                                 subtitle_height_list, canny_img2_list, seq=1,
                                                 output_process=output_process)
    if text_recs is None or len(text_recs) == 0:
        return [], real_img, [], f

    # 送入CRNN检测
    img, f = resize_im(preprocessed_img, scale=Config.SCALE, max_scale=Config.MAX_SCALE)
    result = crnnRec(img, text_recs)

    # 将当前帧的canny2图像写入队列
    if canny_img2_list.__len__() > 0:
        if CANNY_IMG_QUEUE.is_full():
            CANNY_IMG_QUEUE.dequeue()
        CANNY_IMG_QUEUE.enqueue(canny_img2_list)

    # 比较当前帧与上一帧,判断是否为同一字幕,若不是,清空结果列表
    if CANNY_IMG_QUEUE.is_full():
        difference = get_img_difference(CANNY_IMG_QUEUE.queue[0][0], CANNY_IMG_QUEUE.queue[1][0])
        if output_process:
            print("the difference between " + str(imgNo) + " and " + str(imgNo - 1) + ":" + str(difference))
        if difference >= DIFFERENT_THRESHOLD:
            if output_process:
                print(str(imgNo) + " different from " + str(imgNo - 1))
            RESULTS_DICT.clear()

    # 根据是否为同一字幕采用投票策略,输出出现次数最多的结果
    for key in result:
        # 去除检测结果最前端非中文字符,出现重复字符的彻底解决方法应为重新训练
        pattern = re.compile(u"^[^\u4e00-\u9fa5]+")
        result[key][1] = re.sub(pattern, '', result[key][1])
        if result[key][1] in RESULTS_DICT:
            RESULTS_DICT[result[key][1]] = int(RESULTS_DICT[result[key][1]]) + 1
        else:
            RESULTS_DICT[result[key][1]] = 1

    # sorted(d,key=d.__getitem__,reverse=True):
    # results_list = sorted(RESULTS_DICT.items(), key=lambda x: x[1], reverse=True)
    for k in sorted(RESULTS_DICT, key=RESULTS_DICT.__getitem__, reverse=True):
        if output_process:
            print(RESULTS_DICT)
            print("result:" + k + ", times:" + str(RESULTS_DICT[k]))
        result[key][1] = k
        break

    if output_process:
        return result, preprocessed_img, real_recs, f
    else:
        return result, real_img, real_recs, f
