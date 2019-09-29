import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imagehash


class Queue(object):
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


# 在opencv图片中嵌入中文
def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型，若是，需转换为Image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("font/SourceHanSerif.otf", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=font_text)
    # 返回从Image转换得到的cv2图像
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def get_img_difference(img1, img2, hash_type="perception"):
    im1 = Image.fromarray(cv2.cvtColor(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv2.cvtColor(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
    if hash_type == "perception":
        img_hash = imagehash.phash(im1)
        next_hash = imagehash.phash(im2)
    elif hash_type == "average":
        img_hash = imagehash.average_hash(im1)
        next_hash = imagehash.average_hash(im2)
    elif hash_type == "wavelet":
        img_hash = imagehash.whash(im1)
        next_hash = imagehash.whash(im2)
    return img_hash - next_hash


def get_union_coordinate(location_a, location_b):
    x = [location_a[0], location_a[2], location_b[0], location_b[2]]
    y = [location_a[1], location_a[5], location_b[1], location_b[5]]
    union_location = []
    x.sort()
    y.sort()
    if y[1] != location_a[5]:
        for index, value in enumerate(x):
            if value == location_a[0]:
                if x[index+1] != location_a[2]:
                    union_location = [y[0], y[3], x[0], x[3]]
            elif value == location_b[0]:
                if x[index+1] != location_b[2]:
                    union_location = [y[0], y[3], x[0], x[3]]

    return union_location


def draw_ctpn_result_boxes(location_list, img):
    c = (0, 255, 0)

    for index in range(len(location_list)):
        top = location_list[index][0]
        bottom = location_list[index][1]
        left = location_list[index][2]
        right = location_list[index][3]
        cv2.line(img, (left, top), (right, top), c, 2)
        cv2.line(img, (left, top), (left, bottom), c, 2)
        cv2.line(img, (right, bottom), (right, top), c, 2)
        cv2.line(img, (left, bottom), (right, bottom), c, 2)

    return img
