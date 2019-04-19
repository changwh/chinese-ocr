# coding:utf-8
import model
from glob import glob
import numpy as np
from PIL import Image
import time
import cv2


def start(input_path_list, out_path):
    for video_name in input_path_list:
        print(video_name)
        videoCapture = cv2.VideoCapture(video_name)
        success, frame = videoCapture.read()
        frameNum = 0
        while (success):
            frameNum = frameNum + 1
            t = time.time()
            # frame = np.array(frame)
            result, frame, angle = model.model(frame, frameNum, video_name, out_path, model='crnn', detectAngle=False, is_crop=True)
            print("Frame number:{}, It takes time:{}s".format(frameNum, time.time() - t))
            print("---------------------------------------")
            # print("图像的文字朝向为:{}度".format(angle))
            print("识别结果:")

            for key in result:
                print(result[key][1])

            success, frame = videoCapture.read()
    print(out_path)


if __name__ == '__main__':
    # im = Image.open(paths[1])
    # img = np.array(im.convert('RGB'))
    # t = time.time()
    # result,img,angle = model.model(img,model='crnn', detectAngle=False) ## if model == crnn ,you should install pytorch
    # print("It takes time:{}s".format(time.time()-t))
    # print("---------------------------------------")
    # print("图像的文字朝向为:{}度\n".format(angle),"识别结果:\n")
    #
    # for key in result:
    #     print(result[key][1])
    #

    start(["/home/user/PycharmProjects/text-detection-ctpn/data/video/77374694-1-64.flv", "/home/user/PycharmProjects/text-detection-ctpn/data/video/0.flv"],
          "/home/user/test_results")
