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


def start_video(input_path_list, output_path, start_frame=None, end_frame=None, stride=1, output_process=False):
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

        # 输出视频相关
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建视频容器
        videoWriter = cv2.VideoWriter(
            os.path.join(output_path, "{}_results.mp4".format(video_name.split('/')[-1].split('.')[0])),
            cv2.VideoWriter_fourcc(*'mp4v'),
            video_capture.get(cv2.CAP_PROP_FPS),
            (frame_width, frame_height))

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

            result, frame, real_recs, f = model.model(frame, frame_num, video_name, output_path, output_process=output_process)
            print("Frame number:{}, It takes time:{}s".format(frame_num, time.time() - t))
            print("---------------------------------------")
            print("识别结果:")

            for key in result:
                print(result[key][1])

                # 在视频中嵌入识别结果
                frame = cv2_img_add_text(frame, result[key][1], int(result[key][0][0] / f),
                                     int(result[key][0][1] / f) - 120,
                                     text_color=(0, 255, 0), text_size=40)

            # 将加框后图片拼接成视频
            videoWriter.write(frame)
            cv2.imwrite(os.path.join(output_path,
                                     "final_{}_{}.jpg".format(video_name.split('/')[-1].split('.')[0], str(frame_num))),
                        frame)
            success, frame = video_capture.read()
            frame_num += 1

    print(output_path)


if __name__ == '__main__':
    start_video(["/home/user/PycharmProjects/text-detection-ctpn/data/video2/1.mp4"],
                "/home/user/single_test_5",
                start_frame=5500, end_frame=7250, output_process=True)  # news 2-45605 overlap error 2-2722
                # 1: 5500-7250
                # 2: 5325-7250
                # 3: 3875-6275
                # 4: 5275-5925
                # 5: 12825-14300
                # 6: 2625-3450
