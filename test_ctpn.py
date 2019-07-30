import os
import shutil
import sys
import time
import numpy as np
sys.path.append('ctpn')

import cv2

from ctpn.text_detect import test_text_detect

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


def my_ctpn(img, img_no, video_name, output_path):
    real_img = img.copy()

    # ctpn
    text_recs, drawn_img, img, f = test_text_detect(img, top=0.5, bottom=1, left=0, right=1)

    # 获取原图坐标便于预处理
    real_recs = toRealCoordinate(text_recs, f)

    crop_img(real_img, video_name, output_path, real_recs, img_no)

    return drawn_img, real_recs, f


def test_ctpn(input_path_list, output_path, start_frame=None, end_frame=None, stride=1):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for video_name in input_path_list:
        print(video_name)
        video_capture = cv2.VideoCapture(video_name)
        # 根据需要指定开始帧
        if start_frame:                         # 限制了开始帧，从开始帧开始
            frame_num = start_frame
            video_capture.set(1, start_frame)
        else:                                   # 未限制开始帧，从第0帧开始
            frame_num = 0
        success, frame = video_capture.read()

        while success:
            t = time.time()

            # 根据需要指定结束帧
            if end_frame is not None:           # 限制了结束帧，到结束帧停止
                if frame_num > end_frame:
                    video_capture.release()
                    break

            # 根据需要指定步长
            if frame_num % stride != 0:
                success, frame = video_capture.read()
                frame_num += 1
                continue

            drawn_img, real_recs, f = my_ctpn(frame, frame_num, video_name, output_path)

            print("Frame number:{}, It takes time:{}s".format(frame_num, time.time() - t))

            cv2.imwrite(os.path.join(output_path,
                                     "final_{}_{}.jpg".format(video_name.split('/')[-1].split('.')[0], str(frame_num))),
                        drawn_img)
            success, frame = video_capture.read()
            frame_num += 1

    print(output_path)


if __name__ == '__main__':
    test_ctpn(["/home/user/PycharmProjects/text-detection-ctpn/data/news/2.mp4"],
               "/home/user/mytest25",
               start_frame=2000, stride=25)
