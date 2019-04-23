# coding:utf-8
##添加文本方向 检测模型，自动检测文字方向，0、90、180、270
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


def model(img, imgNo, videoName, outputPath, model='keras', adjust=False, detectAngle=False, is_crop=False):
    """
    @@param:img,
    @@param:model,选择的ocr模型，支持keras\pytorch版本
    @@param:adjust 调整文字识别结果
    @@param:detectAngle,是否检测文字朝向
    
    """
    angle = 0
    if detectAngle:

        angle = angle_detect(img=np.copy(img))  ##文字朝向检测
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

    # 字幕过滤
    text_recs = subtitle_filter(text_recs, img.shape[0], img.shape[1])
    if text_recs is None:
        return [], tmp, angle

    # 获取原图坐标便于预处理
    real_recs = toRealCoordinate(text_recs, f)

    if is_crop:
        crop_img(real_img, videoName, outputPath, real_recs, imgNo)

    tmp = real_img.copy()
    preprocessed_img = preprocessing.p_picture(real_recs, tmp, videoName, outputPath, imgNo)
    img, f = resize_im(preprocessed_img, scale=Config.SCALE, max_scale=Config.MAX_SCALE)

    result = crnnRec(img, text_recs, model, adjust=adjust)
    return result, real_img, angle, real_recs, f


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
def subtitle_filter(boxes, img_height, img_width):
    # roi限制
    roi_y_per = 0.7
    roi_x_per = 0.5
    # 宽度限制
    pass
    # 高度限制
    height_filt_per = 0.061  # 0.05
    pass

    temp = []
    for index, box in enumerate(boxes):
        # 将不满足限制条件的文字框加入到待删除列表中
        if box[1] < img_height * roi_y_per or box[0] > img_width * roi_x_per or box[7] - box[1] < img_height * height_filt_per:
            temp.append(index)

    boxes = np.delete(boxes, temp, axis=0)
    return sort_box(boxes)


def toRealCoordinate(text_recs, f):
    tmp = np.zeros((len(text_recs), 8), np.int)

    for index1, text_rec in enumerate(text_recs):
        for index2, point in enumerate(text_rec):
            tmp[index1, index2] = point / f

    return tmp
