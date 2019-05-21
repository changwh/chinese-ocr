# coding:utf-8
import model
from glob import glob
import numpy as np
from PIL import Image
import time
import cv2
import os
import shutil
from PIL import Image, ImageDraw, ImageFont


# 截帧
def getFrame(input_path_list, output_path):
    for video_name in input_path_list:
        print(video_name)
        videoCapture = cv2.VideoCapture(video_name)
        success, frame = videoCapture.read()
        frameNum = 0

        base_name = video_name.split('/')[-1]

        while (success):
            frameNum = frameNum + 1

            # 根据需要跳帧
            if frameNum % 6500 != 0:
                success, frame = videoCapture.read()
                continue
            cv2.imwrite(
                os.path.join(output_path, "origin_{}_{}.jpg".format(base_name.split('.')[0], frameNum)),
                frame)
            success, frame = videoCapture.read()
    print(output_path)


def start_img(input_path_list, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for img_name in input_path_list:
        print(img_name)
        img = cv2.imread(img_name)
        t = time.time()

        result, img, angle, real_recs, f = model.model(img, 0, img_name, output_path, model='crnn', detectAngle=False,
                                                       is_crop=True)
        print("Frame number:{}, It takes time:{}s".format(0, time.time() - t))
        print("---------------------------------------")
        # print("图像的文字朝向为:{}度".format(angle))
        print("识别结果:")

        for key in result:
            print(result[key][1])

            # 在视频中嵌入识别结果
            img = cv2ImgAddText(img, result[key][1], int(result[key][0][0] / f), int(result[key][0][1] / f) - 120,
                                textColor=(0, 255, 0), textSize=50)

        cv2.imwrite(os.path.join(output_path,
                                 "final_{}_{}.jpg".format(img_name.split('/')[-1].split('.')[0], str(0))), img)

    print(output_path)


def start_video(input_path_list, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for video_name in input_path_list:
        print(video_name)
        videoCapture = cv2.VideoCapture(video_name)
        success, frame = videoCapture.read()
        frameNum = 0

        # 输出视频相关
        frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建视频容器
        videoWriter = cv2.VideoWriter(
            os.path.join(output_path, "{}_results.mp4".format(video_name.split('/')[-1].split('.')[0])),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            videoCapture.get(cv2.CAP_PROP_FPS),
            (frame_width, frame_height))

        while (success):
            frameNum = frameNum + 1
            t = time.time()

            # 根据需要跳帧
            if frameNum < 5500 or frameNum > 7250:
                success, frame = videoCapture.read()
                continue

            result, frame, angle, real_recs, f = model.model(frame, frameNum, video_name, output_path, model='crnn',
                                                             detectAngle=False, is_crop=True)
            print("Frame number:{}, It takes time:{}s".format(frameNum, time.time() - t))
            print("---------------------------------------")
            # print("图像的文字朝向为:{}度".format(angle))
            print("识别结果:")

            for key in result:
                print(result[key][1])

                # 在视频中嵌入识别结果
                frame = cv2ImgAddText(frame, result[key][1], int(result[key][0][0] / f),
                                      int(result[key][0][1] / f) - 120,
                                      textColor=(0, 255, 0), textSize=50)

            # 将加框后图片拼接成视频
            videoWriter.write(frame)
            cv2.imwrite(os.path.join(output_path,
                                     "final_{}_{}.jpg".format(video_name.split('/')[-1].split('.')[0], str(frameNum))),
                        frame)
            success, frame = videoCapture.read()

    print(output_path)


# 在opencv图片中嵌入中文
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/SourceHanSerif.otf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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

    start_video(["/home/user/PycharmProjects/text-detection-ctpn/data/video2/1.mp4"],   # 1:5550-7250   2:5330-7250
                "/home/user/test_similarity14")
    # start_img(["/home/user/PycharmProjects/text-detection-ctpn/data/test_img/p1.jpg",
    #            "/home/user/PycharmProjects/text-detection-ctpn/data/test_img/p2.jpg",
    #            "/home/user/PycharmProjects/text-detection-ctpn/data/test_img/p3.jpg",
    #            "/home/user/PycharmProjects/text-detection-ctpn/data/test_img/p4.jpg",
    #            "/home/user/PycharmProjects/text-detection-ctpn/data/test_img/p5.jpg",
    #            "/home/user/PycharmProjects/text-detection-ctpn/data/test_img/p6.jpg"],
    #             "/home/user/test_img_results")   
    # getFrame(["/home/user/PycharmProjects/text-detection-ctpn/data/video2/2.mp4"], "/home/user/frame")