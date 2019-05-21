# -*- coding : UTF-8 -*-

import cv2 as cv
from PIL import Image
import os
import numpy as np
import copy
import imagehash


def openImg_opencv(filename='new.jpg'):
    if os.path.exists(filename):
        image = cv.imread(filename)  # opencv
        return image
    else:
        print("image not found")


def openImg_PIL(filename='new.jpg'):
    if os.path.exists(filename):
        temp = Image.open(filename)  # PIL
        image = np.array(temp)
        return image
    else:
        print("image not found")


def averageGray(image):
    image = image.astype(int)
    for y in range(image.shape[1]):  # y is width
        for x in range(image.shape[0]):  # x is height
            gray = (image[x, y, 0] + image[x, y, 1] + image[x, y, 2]) // 3
            image[x, y] = gray
    return image.astype(np.uint8)


def averageGrayWithWeighted(image):
    image = image.astype(int)
    for y in range(image.shape[1]):  # y is width
        for x in range(image.shape[0]):  # x is height
            gray = image[x, y, 0] * 0.3 + image[x, y, 1] * 0.59 + image[x, y, 2] * 0.11
            image[x, y] = int(gray)
    return image.astype(np.uint8)


def maxGray(image):
    for y in range(image.shape[1]):  # y is width
        for x in range(image.shape[0]):
            gray = max(image[x, y])  # x is height
            image[x, y] = gray
    return image


def resize_opencv(image, weight=128, height=128):
    smallImage = cv.resize(image, (weight, height), interpolation=cv.INTER_LANCZOS4)
    return smallImage


def calculateDifference(image, weight=128, height=128):
    differenceBuffer = []
    for x in range(weight):
        for y in range(height - 1):
            differenceBuffer.append(image[x, y, 0] > image[x, y + 1, 0])
    return differenceBuffer


def makeHash(differ):
    hashOrdString = "0b"
    for value in differ:
        hashOrdString += str(int(value))
    hashString = hex(int(hashOrdString, 2))
    return hashString


def stringToHash(filename='new.jpg'):
    image1 = openImg_opencv(filename)
    grayImage1 = averageGray(copy.deepcopy(image1))
    # plt.imshow(grayImage1)
    # plt.show()
    smallImage1 = resize_opencv(copy.deepcopy(grayImage1))
    # plt.imshow(smallImage1)
    # plt.show()
    differ = calculateDifference(copy.deepcopy(smallImage1))
    return makeHash(differ)


def calculateHammingDistance(differ1, differ2):
    difference = (int(differ1, 16)) ^ (int(differ2, 16))
    return bin(difference).count("1")


def main():
    for i in range(12789, 12831):
        pic1 = stringToHash('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i, i))
        pic2 = stringToHash('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i + 1, i + 1))
        print(str(i) + "and" + str(i + 1) + ",this two picture is " + str(
            (128 * 128 - calculateHammingDistance(pic1, pic2)) / (128 * 128) * 100) + "% similarity")
        if (128 * 128 - calculateHammingDistance(pic1, pic2)) / (128 * 128) < 0.6:
            pic3 = stringToHash('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i + 2, i + 2))
            pic4 = stringToHash('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i - 1, i - 1))
            print(str(i) + "and" + str(i + 2) + ",this two picture is " + str(
                (128 * 128 - calculateHammingDistance(pic1, pic3)) / (
                        128 * 128) * 100) + "% similarity")  # 都小于0.6才认为是两张不同的图
            print(str(i - 1) + "and" + str(i + 1) + ",this two picture is " + str(
                (128 * 128 - calculateHammingDistance(pic4, pic2)) / (
                        128 * 128) * 100) + "% similarity")  # 都小于0.6才认为是两张不同的图
        # todo:若相邻两图相似度较低,重新判断间隔1图的相似度,如果也低,认为是两个不同字幕

    # pic1 = stringToHash('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/4_canny2_5_{}_0.jpg'.format(12823, 12823))
    # pic2 = stringToHash('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/4_canny2_5_{}_0.jpg'.format(12825, 12825))
    # print("this two picture is " + str((16 * 16 - calculateHammingDistance(pic1, pic2)) / (16 * 16) * 100) + "% similarity")


def test():
    for i in range(12789, 12831):
        hash = imagehash.average_hash(
            Image.open('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i, i)))
        nexthash = imagehash.average_hash(
            Image.open('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i + 1, i + 1)))
        print(str(i) + "and" + str(i + 1) + ',' + str(hash - nexthash))
        if (hash - nexthash > 10):
            prehash = imagehash.average_hash(
                Image.open('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i - 1, i - 1)))
            nnexthash = imagehash.average_hash(
                Image.open('/home/user/test_results2_5*5_4*4/cropped_pic_5_{}/5_{}_0.jpg'.format(i + 2, i + 2)))
            print(str(i - 1) + "and" + str(i + 1) + ',' + str(prehash - nexthash))
            print(str(i) + "and" + str(i + 2) + ',' + str(hash - nnexthash))
            if prehash - nexthash > 10 and hash - nnexthash > 10:
                print(str(i) + 'and' + str(i + 1) + 'are different')


def get_img_difference(img1, img2):
    im1 = Image.fromarray(cv.cvtColor(cv.cvtColor(img1, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2RGB))
    im2 = Image.fromarray(cv.cvtColor(cv.cvtColor(img2, cv.COLOR_GRAY2BGR), cv.COLOR_BGR2RGB))
    hash = imagehash.average_hash(im1)
    nexthash = imagehash.average_hash(im2)
    return hash - nexthash


if __name__ == "__main__":
    test()
