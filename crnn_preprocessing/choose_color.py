#coding=utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import heapq
from glob import glob
import copy


# paths = glob('./test/*.*')
# output_dir_86 = 'output_86'
# output_dir_64 = 'output_64'
# separate_output = 'seperate_p'


# # 通过灰度图计算
# def conculate_proportion_v1(img):
#     img = reduce_colors(img)
#     # separate_color(img)
#     gray_image_real = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # cv.imshow('ss', gray_image_real)
#     # if cv.waitKey(0) == ord('q'):
#     #     pass
#
#     row = gray_image_real.shape[0]
#     col = gray_image_real.shape[1]
#
#     row_start = row//8
#     row_end = row_start*7
#     col_start = col//8
#     col_end = col_start*7
#
#     the_list = np.arange(8)
#
#     i=row_start
#     j=col_start
#     while (i < row_end ):
#         j = col_start
#         while(j < col_end):
#             the_list[gray_image_real[i,j]//32] += 1
#
#             j = j+1
#         i = i+1
#     plt.bar(range(len(the_list)), the_list)
#     plt.show()


# # 通过量化降低来计算 kernel = 64， 4个区间
# def conculate_proportion_v2(img ,out_name = 'the_whole_test'):
#     row = img.shape[0]
#     col = img.shape[1]
#     row_start = row//8
#     row_end = row_start*7
#     col_start = col//8
#     col_end = col_start*7
#
#     # col_start = col/10
#     # col_end = col_start*3
#     # col_start = col_start*2
#
#     the_list = np.arange(64)
#
#     #测试：高斯双边滤波
#     # img=cv.bilateralFilter(img,40,75,75)
#
#     img = reduce_colors(img, 64)
#
#     i=row_start
#     j=col_start
#     while (i < row_end ):
#         j = col_start
#         while(j < col_end):
#             x = img[i,j,0]
#             y = img[i,j,1]
#             z = img[i,j,2]
#             index = x*16 + y*4 + z
#             the_list[index] += 1
#             j = j+1
#         i = i+1
#
#     #计算最大元素及其下标
#     # max_num = max(the_list)
#     # max_num_index = np.argmax(the_list)
#
#     Is_bottom = False
#
#     #找出前三个最大元素的下标
#     list2 = the_list.argsort()[-3:][::-1]
#     #显示保存直方图
#     show_the_histogram(the_list,list2,out_name, 4)
#     if the_list[list2[1]]*3 > the_list[list2[0]]*2:
#         Is_bottom = True
#     elif the_list[list2[2]]*3 < the_list[list2[1]]*2:
#         Is_bottom = True
#
#
#     # print Is_bottom
#     if Is_bottom == True:
#         return 0
#     else:
#         return 1
#     # print the_list[list2[0]] ,the_list[list2[1]] ,the_list[list2[2]]


# 通过量化降低来计算 kernel = 86， 3个区间
def conculate_proportion_v3(img, out_name='the_whole_test'):
    # img_procress = copy.deepcopy(img)

    row = img.shape[0]
    col = img.shape[1]
    row_start = row // 8
    row_end = row_start * 7
    col_start = col // 8
    col_end = col_start * 7

    # col_start = col/10
    # col_end = col_start*3
    # col_start = col_start*2

    the_list = np.arange(27)

    #测试：高斯双边滤波
    # img=cv.bilateralFilter(img,40,75,75)

    img = reduce_colors(img, 86)

    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            x = img[i, j, 0]
            y = img[i, j, 1]
            z = img[i, j, 2]
            index = x * 9 + y * 3 + z
            the_list[index] += 1

    #计算最大元素及其下标
    # max_num = max(the_list)
    # max_num_index = np.argmax(the_list)

    is_background = False

    #找出前三个最大元素的下标
    list2 = the_list.argsort()[-4:][::-1]

    #显示保存直方图
    # show_the_histogram(the_list,list2,out_name,3)

    #Ture:有底色
    if the_list[list2[1]] * 3 > the_list[list2[0]] * 2:
        is_background = True
    elif the_list[list2[2]] * 3 < the_list[list2[1]] * 2:
        is_background = True
    elif the_list[list2[3]] * 3 > the_list[list2[2]] * 2:
        is_background = True

    #只保留最大的颜色通道
    # keep_the_largest(img, list2[0], out_name)

    # print Is_bottom
    if is_background == True:
        return 0, img
    else:
        if list2[0] == 26:
            return 2,img
        thresh1 = keep_the_largest(img, list2[0], out_name)
        # cv.imshow('sss', thresh1)
        # if cv.waitKey(0) == ord('q'):
        #     pass
        gray_image_real = cv.cvtColor(thresh1, cv.COLOR_BGR2GRAY)
        return 1, gray_image_real
    # print the_list[list2[0]] ,the_list[list2[1]] ,the_list[list2[2]]


# # 显示保存直方图
# def show_the_histogram(the_list,list2,out_name,index):
#     fig = plt.figure(0)
#     plt.bar(range(len(the_list)), the_list, color = 'deepskyblue')
#     first = turn_to_x(list2[0], index)
#     secend = turn_to_x(list2[1], index)
#     third = turn_to_x(list2[2], index)
#     a = [str(i) for i in first]
#     ls1 = ''.join(a)
#     b = [str(i) for i in secend]
#     ls2 = ''.join(b)
#     c = [str(i) for i in third]
#     ls3 = ''.join(c)
#     output_str = '1: '+ls1 +'    2: ' +ls2 + '     3:'+ls3
#     # print output_str
#     plt.title(output_str,color='blue')
#     if index == 4:
#         # plt.savefig(output_dir_64+'/'+out_name+'_hg_64'+'.jpg')
#         pass
#     if index == 3:
#         # plt.savefig(output_dir_86+'/'+out_name+'_hg_86'+'.jpg')
#         pass
#     plt. close(0)
#     # plt.show()


# # 转化为index进制
# def turn_to_x(num, index):
#     list = np.arange(3)
#     for i in range(3):
#         s = num // index**(2-i)
#         num = num % index**(2-i)
#         list[i] = s
#     return list


def keep_the_largest(img, index, out_name):
    row = img.shape[0]
    col = img.shape[1]

    for i in range(0, row):
        for j in range(0, col):
            x = img[i, j, 0]
            y = img[i, j, 1]
            z = img[i, j, 2]
            index_org = x * 9 + y * 3 + z

            if index_org == index:
                img[i, j] = [255, 255, 255]
            else:
                img[i, j] = [0, 0, 0]

    # cv.imwrite(separate_output + '/' +out_name+'_86.jpg',img)
    return img
    # if cv.waitKey(0) == ord('1'):
    #     pass


# 量化降低
def reduce_colors(img, kernel):
    # kernel = 64 #四个颜色区间  
    # kernel = 86 #三个颜色区间，多一个255/85=3（4）

    img = img // kernel
    # img = img/kernel*kernel + kernel/2

    # cv.imshow('test', img)
    # if cv.waitKey(0) == ord('q'):
    #     pass

    return img


# # rgb颜色直方图
# def calcAndDrawHist(image, color):
#     hist= cv.calcHist([image], [0], None, [256], [0.0,255.0])
#     minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(hist)
#     histImg = np.zeros([256,256,3], np.uint8)
#     hpt = int(0.9* 256);
#
#     for h in range(256):
#         intensity = int(hist[h]*hpt/maxVal)
#         cv.line(histImg,(h,256), (h,256-intensity), color)
#     return histImg
#     # if __name__ == '__main__':
#     # img = cv.resize(src,None,fx=0.6,fy=0.6,interpolation = cv.INTER_CUBIC)
#     # b, g, r = cv.split(img)
#
#     # histImgB = calcAndDrawHist(b, [255, 0, 0])
#     # histImgG = calcAndDrawHist(g, [0, 255, 0])
#     # histImgR = calcAndDrawHist(r, [0, 0, 255])
#
#     # cv.imshow("histImgB", histImgB)
#     # cv.imshow("histImgG", histImgG)
#     # cv.imshow("histImgR", histImgR)
#     # cv.imshow("Img", img)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()


# # 通过hsv分离颜色
# def separate_color(frame):
#     cv.imshow("oringe", frame)
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)                              #色彩空间转换为hsv，便于分离
#
#     lower_hsv_yellow = np.array([11, 43, 46])
#     high_hsv_yellow = np.array([34, 255, 255])
#
#     lower_hsv_blue = np.array([78, 43, 46])
#     high_hsv_blue = np.array([124, 255, 255])
#
#     lower_hsv_white = np.array([0, 0, 200])
#     high_hsv_white = np.array([180, 30, 225])
#
#
#     lower_hsv = lower_hsv_blue                                     #提取颜色的低值
#     high_hsv = high_hsv_blue                                   #提取颜色的高值
#
#     mask = cv.inRange(hsv, lowerb = lower_hsv, upperb = high_hsv)
#     cv.imshow("ending", mask)
#     print(mask.shape)


# # 图片锐化
# def sharpen(img):
#     #自定义卷积核
#     kernel_sharpen_1 = np.array([
#         [-1,-1,-1],
#         [-1,9,-1],
#         [-1,-1,-1]])
#     # kernel_sharpen_2 = np.array([
#     #     [1,1,1],
#     #     [1,-7,1],
#     #     [1,1,1]])
#     # kernel_sharpen_3 = np.array([
#     #     [-1,-1,-1,-1,-1],
#     #     [-1,2,2,2,-1],
#     #     [-1,2,8,2,-1],
#     #     [-1,2,2,2,-1],
#     #     [-1,-1,-1,-1,-1]])/8.0
#     #卷积
#     output_1 = cv.filter2D(img,-1,kernel_sharpen_1)
#     cv.imshow('sharpen_1 Image',output_1)
#     # cv.imwrite('sharpen_1 Image.jpg',output_1)
#     # output_2 = cv.filter2D(image,-1,kernel_sharpen_2)
#     # output_3 = cv.filter2D(image,-1,kernel_sharpen_3)
#     if cv.waitKey(0) == ord('q'):
#         pass
#
#     return output_1


# def glob(pathname):
#     """Return a list of paths matching a pathname pattern.
#
#     The pattern may contain simple shell-style wildcards a la
#     fnmatch. However, unlike fnmatch, filenames starting with a
#     dot are special cases that are not matched by '*' and '?'
#     patterns.
#
#     """
#     return list(iglob(pathname))


# def demo(image):
#     src = cv.imread(image)
#     # src = sharpen(src)
#     out_name = image[-7:-4]
#     check_out = image[-7]
#
#
#     # separate_color(src)
#     # reduce_colors(src)
#     # conculate_proportion_v1(src)
#     # conculate_proportion_v2(src)
#     check_out_value , thresh1= conculate_proportion_v3(src, out_name)  #86
#
#     if check_out == str(check_out_value):
#         print(image +' is correct!')
#         return 1
#     if check_out == '1' and check_out_value == 2:
#         print(image +' is correct!')
#         return 1
#     else:
#         print(image + ' is not correct')
#         return 0


# if __name__ == '__main__':
#     # demo('./test/2202.jpg')
#     accuracy = 0
#     the_number_of_img = 0
#     for index in range(len(paths)):
#         image  = paths[index]
#         # print image
#         accuracy = accuracy+demo(image)
#         the_number_of_img = the_number_of_img+1
#
#     print('\nthe number of the picture is: ' + format(the_number_of_img))
#     print('the accuracy rate of judge is: ' + format(float(accuracy)/float(the_number_of_img),'.2f'))
