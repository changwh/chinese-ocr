# coding:utf-8
import model
import numpy as np
import time
import cv2
import os
import shutil
from PIL import Image, ImageDraw, ImageFont


# 在opencv图片中嵌入中文
def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("font/SourceHanSerif.otf", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=font_text)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 从图片读取 TODO:model修改后可能需要调试
def start_img(input_path_list, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for img_name in input_path_list:
        print(img_name)
        img = cv2.imread(img_name)
        t = time.time()

        result, img, real_recs, f = model.model(img, 0, img_name, output_path, model='crnn', output_process=True)
        print("Frame number:{}, It takes time:{}s".format(0, time.time() - t))
        print("---------------------------------------")
        print("识别结果:")

        for key in result:
            print(result[key][1])

            # 在视频中嵌入识别结果
            img = cv2_img_add_text(img, result[key][1], int(result[key][0][0] / f), int(result[key][0][1] / f) - 120,
                                   text_color=(0, 255, 0), text_size=50)

    print(output_path)


def start_video(input_path_list, output_path, start_frame=None, end_frame=None, output_process=False):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for video_name in input_path_list:
        print(video_name)
        video_capture = cv2.VideoCapture(video_name)
        success, frame = video_capture.read()
        frame_num = 0

        # 输出视频相关
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建视频容器
        videoWriter = cv2.VideoWriter(
            os.path.join(output_path, "{}_results.mp4".format(video_name.split('/')[-1].split('.')[0])),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            video_capture.get(cv2.CAP_PROP_FPS),
            (frame_width, frame_height))

        while (success):
            frame_num = frame_num + 1
            t = time.time()

            # 根据需要跳帧
            if start_frame is not None and end_frame is None:           # 只限制起始帧
                if frame_num < start_frame:
                    success, frame = video_capture.read()
                    continue
            elif start_frame is None and end_frame is not None:         # 只限制结束帧
                if frame_num > end_frame:
                    success, frame = video_capture.read()
                    continue
            elif start_frame is not None and end_frame is not None:     # 限制起始帧和结束帧
                if frame_num < start_frame or frame_num > end_frame:
                    success, frame = video_capture.read()
                    continue

            result, frame, real_recs, f = model.model(frame, frame_num, video_name, output_path,
                                                      model='crnn', output_process=output_process)
            print("Frame number:{}, It takes time:{}s".format(frame_num, time.time() - t))
            print("---------------------------------------")
            print("识别结果:")

            for key in result:
                print(result[key][1])

                # 在视频中嵌入识别结果
                frame = cv2_img_add_text(frame, result[key][1], int(result[key][0][0] / f),
                                         int(result[key][0][1] / f) - 120,
                                         text_color=(0, 255, 0), text_size=50)

            # 将加框后图片拼接成视频
            videoWriter.write(frame)
            cv2.imwrite(os.path.join(output_path,
                                     "final_{}_{}.jpg".format(video_name.split('/')[-1].split('.')[0], str(frame_num))),
                        frame)
            success, frame = video_capture.read()

    print(output_path)


if __name__ == '__main__':
    start_video(["/home/user/PycharmProjects/text-detection-ctpn/data/video2/1.mp4"],
                "/home/user/test_similarity14",
                start_frame=5500, end_frame=7250, output_process=True)
