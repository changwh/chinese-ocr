# -*- coding : UTF-8 -*-

import cv2 as cv
from PIL import Image
import imagehash


def test():
    for i in range(12789, 12831):
        img_hash = imagehash.average_hash(
            Image.open('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i, i)))
        next_hash = imagehash.average_hash(
            Image.open('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i + 1, i + 1)))
        print(str(i) + "and" + str(i + 1) + ',' + str(img_hash - next_hash))
        if img_hash - next_hash > 10:
            pre_hash = imagehash.average_hash(
                Image.open('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i - 1, i - 1)))
            after_next_hash = imagehash.average_hash(
                Image.open('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i + 2, i + 2)))
            print(str(i - 1) + "and" + str(i + 1) + ',' + str(pre_hash - next_hash))
            print(str(i) + "and" + str(i + 2) + ',' + str(img_hash - after_next_hash))
            if pre_hash - next_hash > 10 and img_hash - after_next_hash > 10:
                print(str(i) + 'and' + str(i + 1) + 'are different')


def get_img_difference(img1, img2):
    im1 = Image.fromarray(cv.cvtColor(cv.cvtColor(img1, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(cv.cvtColor(img2, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2RGB))
    img_hash = imagehash.average_hash(im1)
    next_hash = imagehash.average_hash(im2)
    return img_hash - next_hash


if __name__ == "__main__":
    test()
