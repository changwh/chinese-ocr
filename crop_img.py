import os
import shutil
import cv2


def crop_img(img, video_name, output_path, boxes, frameNum):
    base_name = video_name.split('/')[-1]

    # 对每一帧分别创建一个文件夹存放截取出的字幕
    if os.path.exists(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum))):
        shutil.rmtree(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum)))
    os.makedirs(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum)))

    i = 0

    for box in boxes:
        # point = box.split(',')
        cropped = img[int(box[1]):int(box[7]), int(box[0]):int(box[6])]  # 高度、宽度
        cv2.imwrite(os.path.join(output_path, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                                 "{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, str(i))), cropped)
        i = i + 1
