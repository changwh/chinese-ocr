  #coding=utf-8
import numpy as np
import random
# import time
import cv2 as cv
import copy
import os
import sys
import natsort
from .clean_image import wash_canny_picture
from .news_function import news_procress
from .choose_color import conculate_proportion_v3
# np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)

from matplotlib import pyplot as plt

def dilate_demo(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))   #kernel(5, 5)
    dst = cv.dilate(thresh, kernel)
    # cv.imshow('dilate image', dst)
    return dst

def precess(thresh1, candy_img):
    row = thresh1.shape[0]
    col = thresh1.shape[1]
    # print(row, col)
    i=0
    j=0
    while (i < row ):
        j = 0
        while(j < col):
            if (thresh1[i, j] == 255) and (candy_img [i, j, 0] == 255):
                candy_img[i, j] = [255,255,255]
            else:
                candy_img[i, j] = [0, 0, 0]
            j = j+1
        i = i+1
    return candy_img

def precess_2(img):
    row = img.shape[0]
    col = img.shape[1]
    i = 0
    j = 0
    while(j < col):
        if img[0, j, 0] == 255:
            # global line_extent
            # line_extent = 0
            clean_the_line(img, 1, j, row, col)
            # print(line_extent)
        j = j+1

    i = 0
    j = 0
    while(j < col):
        if img[row-1 , j, 0] == 255:
            # global line_extent
            # line_extent = 0
            clean_the_line(img,row-2, j,row, col)
            # print(line_extent)
        j = j+1
    i = 0
    j = 0
    while(i < row):
        if img[i, j, 0] == 255:
            # global line_extent
            # line_extent = 0
            clean_the_line(img,i, j,row, col)
            # print(line_extent)
        i = i+1


    return img

def clean_the_line(img, i, j,row,col):
    img[i,j] = [0, 0, 0]
    # global line_extent
    # line_extent = line_extent + 1
    # if line_extent>100:
    #   return
    # print(i, j)
    if img[i, j+1, 0] == 255 and j != col-2:      #右边
        clean_the_line(img, i, j+1, row,col)
    if img[i+1, j+1, 0] == 255 and j != col-2 and i != row-2:    #右下
        clean_the_line(img, i+1, j+1,row,col)
    if img[i+1, j, 0] == 255 and i != row-2:      #下边
        clean_the_line(img, i+1, j, row,col)
    if img[i+1, j-1, 0] ==255 and i != row-2 and j != 1:       #左下
        clean_the_line(img, i+1, j-1, row,col)
    if img[i, j-1, 0] == 255 and j != 1:      #左边
        clean_the_line(img, i, j-1,row,col)
    if img[i-1, j+1, 0] == 255 and j != col-2 and i != 1:    #右上
        clean_the_line(img, i-1, j+1, row,col)
    if img[i-1, j, 0] == 255 and i != 1:      #上边
        clean_the_line(img, i-1, j, row,col)
    if img[i-1, j-1, 0] == 255 and i != 1 and j != 1:      #左上
        clean_the_line(img, i-1, j-1,row,col)
    return 0

def dilate_demo2(img,dilate_kernel):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #cv.imshow('binary iamge', thresh)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_kernel, dilate_kernel))   #kernel(8, 8)
    dst = cv.dilate(thresh, kernel)
    #cv.imwrite('pengzhang.png', dst)
   # cv.imshow('dilate image', dst)
    return dst

def erode_demo(img,erode_kernel):
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (erode_kernel , erode_kernel))    #kernel(5,5)
    dst = cv.erode(thresh, kernel=kernel)
    #cv.imwrite('jieguo.png', dst)
    #cv.imshow('erode_demo', dst)
    return dst

def get_result(step2, binary_img):
	row = binary_img.shape[0]
	col = binary_img.shape[1]
	i=0
	j=0
	while (i < row ):
		j = 0
		while(j < col):
			if step2[i, j] == 0:
				binary_img [i, j] = [0, 0, 0]
			j = j+1
		i = i+1
	return binary_img

def fit(i, j, img):
	ff = 200
	img[i, j, 2] = ff
	img[i, j+1, 2] = ff
	img[i, j-1, 2] = ff
	img[i+1, j, 2] = ff
	img[i-1, j, 2] = ff
	img[i+1, j+1, 2] = ff
	img[i-1, j-1, 2] = ff
	img[i+1, j-1, 2] = ff
	img[i-1, j+1, 2] = ff

def convert_demo(image):
    height, width, channels = image.shape
    print("width:%s,height:%s,channels:%s" % (width, height, channels))

    for row in range(height):
        for list in range(width):
            for c in range(channels):
                pv = image[row, list, c]
                image[row, list, c] = 255 - pv
    #cv.imshow("AfterDeal", image)

def convert(image):
    # image = cv.imread(image,0)
    height = image.shape[0]
    width = image.shape[1]
    for i in range(height):
        for j in range(width):
            image[i,j] = 255 - image[i,j]
    return image

def get_the_width(gray_image, candy_img2):
    ret,thresh1 = cv.threshold(gray_image,245,255,cv.THRESH_BINARY)
    # cv.imshow('1', thresh1)
    # cv.imwrite('2.jpg', candy_img2)
    row = thresh1.shape[0]
    col = thresh1.shape[1]
    i = 0
    j = 0
    num = 0
    list1 = []
    while (i < row ):
        j = 0
        while(j < col):
            if thresh1[i, j] == 255:
                num = num + 1
                list1.append([i,j])
            j = j+1
        i = i+1
    v = 0

    index = 0
    if num > 0:
        the_width_list = np.arange(50)
        while v<50 and index < 300:
            the_width = creat_random(candy_img2,num,list1)
            if the_width != 0:
                the_width_list[v] = the_width
                v = v+1
            index = index + 1

        counts = np.bincount(the_width_list)
        final_width = np.argmax(counts)
        y = 0
        for x in the_width_list:
            y = x+y
        if y>500:
            return 30
        return final_width
    return 0

def creat_random(candy_img2, num,list1):
    ss = random.randint(0,num-1)
    max_x = candy_img2.shape[0]
    max_y = candy_img2.shape[1]

    x = list1[ss][0]
    y = list1[ss][1]
    top = 0
    top_index = 0
    if candy_img2[x, y, 0] == 0:
        while(top_index == 0):
            x = x+1
            if x < max_x:
                if candy_img2[x, y, 0] == 0:
                    top = top+1
                else:
                    top_index = 1
            else:
                top_index = 1

    x = list1[ss][0]
    y = list1[ss][1]
    bottom = 0
    bottom_index = 0
    if candy_img2[x, y, 0] == 0:
        while(bottom_index == 0):
            x = x-1
            if x >= 0:
                if candy_img2[x, y, 0] == 0:
                    bottom = bottom+1
                else:
                    bottom_index = 1
            else:
                bottom_index = 1

    x = list1[ss][0]
    y = list1[ss][1]
    left = 0
    left_index = 0
    if candy_img2[x, y, 0] == 0:
        while(left_index == 0):
            y = y-1
            if y >= 0:
                if candy_img2[x, y, 0] == 0:
                    left = left+1
                else:
                    left_index = 1
            else:
                left_index = 1

    x = list1[ss][0]
    y = list1[ss][1]
    right = 0
    right_index = 0
    if candy_img2[x, y, 0] == 0:
        while(right_index == 0):
            y = y + 1
            if y < max_y:
                if candy_img2[x, y, 0] == 0:
                    right = right+1
                else:
                  right_index = 1
            else:
                right_index = 1


    the_num1 = top + bottom
    the_num2 = left + right

    if the_num1<the_num2:
        the_final_num = the_num1
    else:
        the_final_num = the_num2


    if the_final_num < 10 and the_final_num > 1:
        # print the_final_num + 1
        return the_final_num + 1

    # print top , bottom
    # print left, right

    return 0

def main(left, top, right, bottom, img, videoName, outputPath, frameNum, index):


    # left = 680
    # top = 600
    # right = 1100
    # bottom = 660w
    # picture_name = '900'
    base_name = videoName.split('/')[-1]
    # start = time.time()
    # img = cv.imread( picture_name+'.jpg')
    roi_real = img[top:bottom, left:right]
    roi = copy.deepcopy(roi_real)


    #读取灰度图
    gray_image_real = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray_image = copy.deepcopy(gray_image_real)
    # cv.imshow('gray_image', gray_image)

    check_out_value, thresh_cc = conculate_proportion_v3(roi_real)  #0.4

    picture_name = str(base_name)+'_'+str(frameNum)+'_'+str(index)

    # 边缘检测得到边缘图片
    canny_img = cv.Canny(gray_image, 40, 120)  # 40 120
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "1_canny_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img)

    # if '2' == str(check_out_value):
    #     print(picture_name +' is without background color! And it is white')
    #
    # if '1' == str(check_out_value):
    #     print(picture_name +' is without background color!And it is not white')

    if '0' == str(check_out_value):
        # print(picture_name + ' is with background color!')
        im_at_mean = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 10)
        # cv.imshow('ss',im_at_mean)
        # if cv.waitKey(0) ==ord('q'):
        #     pass

        return roi_real, -1, canny_img


    # candy_img = cv.Canny(roi, 40, 120)  #40 120
    # candy_img = cv.Canny(gray_image, 40, 120)  #40 120
    # cv.imshow('start', candy_img)
    # cv.imwrite('fotk.jpg',candy_img)

    # if cv.waitKey(0) ==ord('q'):
    #     pass
    # cv.imwrite('start.jpg', candy_img)


    #是否需要分离
    # Is_roll = True
    Is_roll = False
    if (Is_roll == True):
        img2_0, line_num = news_procress(canny_img, roi)
        # left = left+line_num
        return 0


    if '1' == str(check_out_value):
        thresh1 = thresh_cc
    if '2' == str(check_out_value):
        #阈值处理
        ret,thresh1 = cv.threshold(gray_image,200,255,cv.THRESH_BINARY)
    # cv.imshow('thresh1!', thresh1)
    # cv.imwrite('mm.jpg', thresh1)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "2_thresh1_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), thresh1)

    #膨胀
    thresh2 = dilate_demo(thresh1)
    # cv.imshow('dilate_demo', thresh2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "3_dilate_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), thresh2)

    canny_img = cv.cvtColor(canny_img, cv.COLOR_GRAY2BGR)
    canny_img2 = precess(thresh2,canny_img)  #0.2s


    # candy_img2 = precess_2(candy_img2)
    # cv.imshow('gray_image', gray_image)
    # cv.imshow('candy 2-1', candy_img2)
    # print gray_image

    if '1' == str(check_out_value):
        gray_image = erode_demo(thresh_cc, 3)
        # cv.imshow('thresh1!', gray_image)

    the_width = get_the_width(gray_image, canny_img2) #0.25s


    #去除上下两边的canny噪声
    canny_img2, subtitle_height = wash_canny_picture(canny_img2)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "4_canny2_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), canny_img2)

    # cv.imshow('candy 2-2', candy_img2)
    # cv.imwrite(picture_name+'candy 2-2.jpg', candy_img2)

    #先判断是否正确，the_width=30说明预知检测出现了问题，需要反色
    if the_width == 30:
        convert(gray_image)
        # cv.imshow('ss',gray_image)
        # the_width = get_the_width(gray_image, candy_img2)

    # print('The width is: ' + str(the_width))

    # TODO:需要通过测试获得更多的参数
    if the_width >= 6:
        dilate_kernel = 12
        erode_kernel = 10
    if the_width >= 4 and the_width < 6:
        dilate_kernel = 7
        erode_kernel = 5
    elif the_width <= 2:
        dilate_kernel = 4
        erode_kernel = 3
    elif the_width > 2 and the_width < 4:
        dilate_kernel = 6
        erode_kernel = 4

    step1 = dilate_demo2(canny_img2, dilate_kernel)
    step2 = erode_demo(step1, erode_kernel)
    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "5_erode_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), step2)

    # cv.imshow('step1', step1)
    # cv.imshow('step2', step2)
    # cv.imwrite('ss.jpg',step2)

    output = get_result(step2, roi)
    # cv.imshow('final', output)

    convert(output) #0.09

    cv.imwrite(os.path.join(outputPath, "cropped_pic_{}_{}".format(base_name.split('.')[0], frameNum),
                            "6_output_{}_{}_{}.jpg".format(base_name.split('.')[0], frameNum, index)), output)

    # cv.imwrite('final.jpg',output)
    # img[top:bottom, left:right] = output
    # cv.imshow('final!!!!!', img)
    # cv.imwrite(picture_name+'-final.jpg', img)
    # end = (time.time() - start)
    # print("Time used:",end)

    # if cv.waitKey(0) ==ord('q'):
    #     pass
    return output, subtitle_height, canny_img2


if __name__ == '__main__':

    # left = 400
    # top = 620
    # right = 870
    # bottom = 680
    left = 0
    top = 0
    right = 740
    bottom = 136
    picture_name = '109'

    main(left, top, right, bottom, picture_name)

