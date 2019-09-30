# coding:utf-8
import model
import time
import cv2
import os
import shutil
from utils import cv2_img_add_text
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="news subtitle ocr")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="The input file to be processed",
        default="/home/user/PycharmProjects/text-detection-ctpn/data/news/1.mp4"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        nargs="?",
        help="The output path to store frame images",
        default="/home/user/news_test"
    )
    parser.add_argument(
        "-sf",
        "--start_frame",
        type=int,
        nargs="?",
        help="Define which frame the program starts from",
        default=4000
    )
    parser.add_argument(
        "-ef",
        "--end_frame",
        type=int,
        nargs="?",
        help="Define which frame the program ends with",
        default=7200
    )
    parser.add_argument(
        "-s",
        "--stride",
        type=int,
        nargs="?",
        help="Define how many frames will be skip each time",
        default=1
    )
    parser.add_argument(
        "-op",
        "--output_process",
        action="store_true",
        help="output all the process result",
        default=True  # todo:False
    )
    return parser.parse_args()


def start_video(input_path_list, output_path, start_frame, end_frame, stride, output_process):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for video_name in input_path_list:
        print(video_name)

        video_capture = cv2.VideoCapture(video_name)
        frame_num = start_frame
        # 指定开始帧
        video_capture.set(1, start_frame)
        success, frame = video_capture.read()

        # 输出视频相关
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建视频容器
        video_writer = cv2.VideoWriter(
            os.path.join(output_path, "{}_results.mp4".format(video_name.split('/')[-1].split('.')[0])),
            cv2.VideoWriter_fourcc(*'mp4v'),
            video_capture.get(cv2.CAP_PROP_FPS),
            (frame_width, frame_height))

        while success:
            t = time.time()

            # 根据需要指定结束帧
            if end_frame != 0:  # 限制了结束帧，到结束帧停止
                if frame_num > end_frame:
                    video_capture.release()
                    break

            result, frame, is_scroll, ratio = model.model_news(frame, frame_num, video_name, output_path, output_process=output_process)
            print("Frame number:{}, It takes time:{}s".format(frame_num, time.time() - t))
            print("---------------------------------------")
            print("识别结果:")

            for key in result:
                print(result[key][1])

                # 在视频中嵌入识别结果
                if is_scroll[key] is False:
                    frame = cv2_img_add_text(frame, result[key][1], int(result[key][0][0] / ratio),
                                             int(result[key][0][1] / ratio) - 120,
                                             text_color=(0, 255, 0), text_size=40)
                else:
                    frame = cv2_img_add_text(frame, result[key][1], int(result[key][0][0] / ratio),
                                             int(result[key][0][1] / ratio) - 120,
                                             text_color=(255, 0, 0), text_size=40)

            # 将加框后图片拼接成视频
            video_writer.write(frame)
            cv2.imwrite(os.path.join(output_path,
                                     "final_{}_{}.jpg".format(video_name.split('/')[-1].split('.')[0], str(frame_num))),
                        frame)

            # 根据需要指定步长
            frame_num += stride
            video_capture.set(1, frame_num)

            success, frame = video_capture.read()

    print(output_path)


def main():
    args = parse_arguments()

    input_file_list = args.input_file.split(",")
    output_path = args.output_path
    start_frame = args.start_frame
    end_frame = args.end_frame
    stride = args.stride
    output_process = args.output_process

    start_video(input_file_list, output_path, start_frame=start_frame, end_frame=end_frame, stride=stride,
                output_process=output_process)


if __name__ == '__main__':
    main()
    # 1: 5500-7250
    # 2: 5325-7250
    # 3: 3875-6275
    # 4: 5275-5925
    # 5: 12825-14300
    # 6: 2625-3450
