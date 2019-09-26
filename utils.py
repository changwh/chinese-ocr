import os
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


def show_gray_img(images_dir):
    image_list = [x for x in os.listdir(images_dir) if x.endswith(("jpg", "jpeg", "png", "bmp"))]

    for image_name in image_list:
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)
        image_convert = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        w = int(image.shape[1] / (image.shape[0] * 1.0 / 32))
        image = cv2.resize(image, (w, 32))
        image_convert = cv2.resize(image_convert, (w, 32))
        cv2.imshow(image_name, image)
        cv2.imshow(image_name+"_convert", image_convert)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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


# def test_whash():
#     for i in range(5536, 5760):
#         im1 = Image.open('/home/user/single_test_5/cropped_pic_1_{}/4_canny2_1_{}_0.jpg'.format(i, i))
#         im2 = Image.open('/home/user/single_test_5/cropped_pic_1_{}/4_canny2_1_{}_0.jpg'.format(i + 1, i + 1))
#
#         img_hash = imagehash.whash(im1)
#         next_hash = imagehash.whash(im2)
#
#         print(str(i) + "and" + str(i + 1) + ',' + str(img_hash - next_hash))
#         if img_hash - next_hash > 10:
#             im3 = Image.open('/home/user/single_test_5/cropped_pic_1_{}/4_canny2_1_{}_0.jpg'.format(i - 1, i - 1))
#             im4 = Image.open('/home/user/single_test_5/cropped_pic_1_{}/4_canny2_1_{}_0.jpg'.format(i + 2, i + 2))
#
#             pre_hash = imagehash.whash(im3)
#             after_next_hash = imagehash.whash(im4)
#
#             print(str(i - 1) + "and" + str(i + 1) + ',' + str(pre_hash - next_hash))
#             print(str(i) + "and" + str(i + 2) + ',' + str(img_hash - after_next_hash))
#             if pre_hash - next_hash > 10 and img_hash - after_next_hash > 10:
#                 print(str(i) + 'and' + str(i + 1) + 'are different')


if __name__ == '__main__':
    # main('/home/user/桌面/gray')
    # main('/home/user/PycharmProjects/my_crnn_trainer/my_data_generator/out/train')
    pass
