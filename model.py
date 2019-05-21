# coding:utf-8
##添加文本方向 检测模型，自动检测文字方向，0、90、180、270
import os
import sys

sys.path.append('ctpn')

from ctpn.text_detect import text_detect
from crop_img import crop_img

from ocr.model import predict as ocr
from angle.predict import predict as angle_detect  ##文字方向检测
from crnn.crnn import crnnOcr
from math import *
import numpy as np
import cv2
from PIL import Image
from crnn_preprocessing import preprocessing
from ctpn.ctpn.cfg import Config
from ctpn.ctpn.other import resize_im
from crnn_preprocessing.img_difference import get_img_difference
from random import randint


class Queue():

    def __init__(self, size):
        self.size = size
        self.front = -1
        self.rear = -1
        self.queue = []

    def enqueue(self, ele):  # 入队操作
        if self.isfull():
            # raise exception("queue is full")
            pass
        else:
            self.queue.append(ele)
            self.rear = self.rear + 1

    def dequeue(self):  # 出队操作
        if self.isempty():
            # raise exception("queue is empty")
            pass
        else:
            self.queue.pop(0)
            self.front = self.front + 1

    def isfull(self):
        return self.rear - self.front == self.size

    def isempty(self):
        return self.front == self.rear

    def emptyqueue(self):
        self.rear = self.front = -1
        self.queue = []

    def showQueue(self):
        print(self.queue)


# global variables
canny_img_queue = Queue(2)
height_filt_per = 0.04
left_top_y = 0.7
left_bottom_y = 1
count_of_frame_with_subtitle = 0
results_dict = {}


def crnnRec(im, text_recs, ocrMode='keras', adjust=False):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box

    """
    index = 0
    results = {}
    xDim, yDim = im.shape[1], im.shape[0]

    for index, rec in enumerate(text_recs):
        results[index] = [rec, ]
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  ##图像倾斜角度

        partImg = dumpRotateImage(im, degree, pt1, pt2, pt3, pt4)

        image = Image.fromarray(partImg).convert('L')
        if ocrMode == 'keras':
            sim_pred = ocr(image)
        else:
            sim_pred = crnnOcr(image)

        results[index].append(sim_pred)  ##识别文字

    return results


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
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

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


# 字幕过滤
def subtitle_filter1(boxes, img_height, img_width):
    # roi限制
    roi_y_per = 0.7
    roi_x_per = 0.5
    # 宽度限制
    pass
    # 高度限制
    # height_filt_per = 0.061  # 0.05
    pass

    temp = []
    for index, box in enumerate(boxes):
        # 将不满足限制条件的文字框加入到待删除列表中
        if box[1] < img_height * roi_y_per or \
                box[0] > img_width * roi_x_per:
            temp.append(index)

    boxes = np.delete(boxes, temp, axis=0)
    return sort_box(boxes)


# 字幕过滤
def subtitle_filter2(boxes, img_height, img_width, real_height, real_width, height_filt_per, left_top_y, left_bottom_y, subtitle_height_list, canny_img2_list):
    # 宽度限制
    pass
    # 高度限制
    # height_filt_per = 0.061  # 0.05
    pass

    temp = []

    # TODO:将判断条件改写成绝对值
    for index, subtitle_height in enumerate(subtitle_height_list):
        # 将不满足限制条件的文字框加入到待删除列表中
        if (subtitle_height < real_height * (height_filt_per - 0.01) or subtitle_height > real_height * (height_filt_per + 0.03)) and count_of_frame_with_subtitle < 500:
            temp.append(index)
        if (subtitle_height < real_height * (height_filt_per - 0.005) or subtitle_height > real_height * (height_filt_per + 0.005)) and count_of_frame_with_subtitle >= 500:
            # print("filter 1")
            temp.append(index)
        if (boxes[index][1] < (left_top_y - 0.015) * img_height or boxes[index][7] > (left_bottom_y + 0.015) * img_height) and count_of_frame_with_subtitle >= 500:
            # print("filter 2")
            temp.append(index)

    boxes = np.delete(boxes, temp, axis=0)
    # a = [a[i] for i in range(len(a)) if (i not in b)]
    canny_img2_list = [canny_img2_list[i] for i in range(len(canny_img2_list)) if (i not in temp)]
    return boxes, canny_img2_list


# 获取原始尺寸下的坐标
def toRealCoordinate(text_recs, f):
    tmp = np.zeros((len(text_recs), 8), np.int)

    for index1, text_rec in enumerate(text_recs):
        for index2, point in enumerate(text_rec):
            tmp[index1, index2] = point / f

    return tmp


def model(img, imgNo, videoName, outputPath, model='keras', adjust=False, detectAngle=False, is_crop=False):
    """
    @@param:img,
    @@param:model,选择的ocr模型，支持keras\\pytorch版本
    @@param:adjust 调整文字识别结果
    @@param:detectAngle,是否检测文字朝向
    """
    angle = 0
    if detectAngle:

        angle = angle_detect(img=np.copy(img))  # 文字朝向检测
        im = Image.fromarray(img)
        if angle == 90:
            im = im.transpose(Image.ROTATE_90)
        elif angle == 180:
            im = im.transpose(Image.ROTATE_180)
        elif angle == 270:
            im = im.transpose(Image.ROTATE_270)
        img = np.array(im)

    real_img = img.copy()

    text_recs, tmp, img, f = text_detect(img)

    # 第一次字幕过滤(位置信息)
    text_recs = subtitle_filter1(text_recs, img.shape[0], img.shape[1])
    if text_recs is None or len(text_recs) == 0:
        return [], real_img, angle, [], f

    # 获取原图坐标便于预处理
    real_recs = toRealCoordinate(text_recs, f)

    if is_crop:
        crop_img(real_img, videoName, outputPath, real_recs, imgNo)

    tmp = real_img.copy()
    preprocessed_img, subtitle_height_list, canny_img2_list = preprocessing.p_picture(real_recs, tmp, videoName,
                                                                                      outputPath,
                                                                                      imgNo)
    np.savetxt(
        os.path.join(outputPath, "subtitle_height_{}_{}.txt".format(videoName.split('/')[-1].split('.')[0], imgNo)),
        subtitle_height_list, fmt='%d')

    # 每一帧中只随机抽取一个高度,避免某一帧中非字幕文本对平均高度的影响,通过前500帧含有字幕的图像获取字幕高度
    global height_filt_per
    global left_top_y
    global left_bottom_y
    global count_of_frame_with_subtitle
    global results_dict
    if subtitle_height_list and count_of_frame_with_subtitle < 500:
        index = randint(0, len(subtitle_height_list) - 1)
        height_filt_per = 0.9 * height_filt_per + 0.1 * subtitle_height_list[index] / real_img.shape[0]
        left_top_y = 0.9 * left_top_y + 0.1 * real_recs[index][1] / real_img.shape[0]
        left_bottom_y = 0.9 * left_bottom_y + 0.1 * real_recs[index][7] / real_img.shape[0]
        count_of_frame_with_subtitle = count_of_frame_with_subtitle + 1


    # 第二次字幕过滤(高度,更精确的横坐标)
    text_recs, canny_img2_list = subtitle_filter2(text_recs, img.shape[0], img.shape[1], real_img.shape[0], real_img.shape[1], height_filt_per, left_top_y, left_bottom_y, subtitle_height_list, canny_img2_list)
    if text_recs is None or len(text_recs) == 0:
        return [], real_img, angle, [], f

    # 送入CRNN检测
    img, f = resize_im(preprocessed_img, scale=Config.SCALE, max_scale=Config.MAX_SCALE)
    result = crnnRec(img, text_recs, model, adjust=adjust)

    # 将当前帧的canny2图像写入队列
    if canny_img2_list.__len__() > 0:
        if canny_img_queue.isfull():
            canny_img_queue.dequeue()
        canny_img_queue.enqueue(canny_img2_list)
        # canny_img_queue.showQueue()

    # 比较当前帧与上一帧,判断是否为统一字幕,若不是,清空结果列表
    if canny_img_queue.isfull():
        difference = get_img_difference(canny_img_queue.queue[0][0], canny_img_queue.queue[1][0])
        print("the difference between " + str(imgNo) + " and " + str(imgNo - 1) + ":" + str(difference))
        if difference >= 6:
            print(str(imgNo) + " different from " + str(imgNo - 1))
            results_dict.clear()

    # 根据是否为同一字幕采用投票策略,输出出现最多的结果
    for key in result:
        if result[key][1] in results_dict:
            results_dict[result[key][1]] = int(results_dict[result[key][1]]) + 1
        else:
            results_dict[result[key][1]] = 1

    results_list = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    print(results_list[0])

    # TODO:增大筛选力度,否则需要考虑同一帧中出现多个疑似字幕的情况
    return result, preprocessed_img, angle, real_recs, f
