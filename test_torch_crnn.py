# coding:utf-8
from PIL import Image
from PIL import ImageOps
from glob import glob

import time
import sys
sys.path.insert(1, "./crnn")

import torch
import torch.utils.data
from torch.autograd import Variable
import crnn.util as util
import crnn.dataset as dataset
import crnn.models.crnn as crnn
import crnn.keys as keys
import cv2
import os

GPU = True


def crnnSource():
    alphabet = keys.alphabet
    # alphabet = keys.alphabet2   # 另一个项目的
    converter = util.strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
    else:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cpu()
    # path = './samples/model_acc97.pth'  # pycharm
    # path = './samples/mixed_second_finetune_acc97p7.pth'  # 另一个项目的
    path = './samples/crnn_Rec_done_5_3334.pth'  # 另一个项目的
    # path = 'crnn/samples/model_acc97.pth'   # vscode
    model.eval()
    model.load_state_dict(torch.load(path))
    return model, converter


def crnnOcr(image):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im
    @@text_recs:text box

    """

    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    # print "im size:{},{}".format(image.size,w)
    transformer = dataset.resizeNormalize((w, 32))
    if torch.cuda.is_available() and GPU:
        image = transformer(image).cuda()
    else:
        image = transformer(image).cpu()

    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    if len(sim_pred) > 0:
        if sim_pred[0] == u'-':
            sim_pred = sim_pred[1:]

    return sim_pred


def get_crnn_result(gray_im):
    sim_pred = crnnOcr(gray_im)

    return sim_pred


def main(input_dir):
    paths = sorted(glob(input_dir + '/*.*'))
    for img_name in paths:
        print(img_name)
        img = Image.open(img_name).convert('L')
        t = time.time()
        result1 = get_crnn_result(img)
        gray_im = ImageOps.invert(img)
        result2 = get_crnn_result(gray_im)

        print("Frame number:{}, It takes time:{}s".format(0, time.time() - t))
        print("---------------------------------------")
        print("识别结果:")
        print("origin:", result1)
        print("invert:", result2)

# 切割存在两种字体的文字框图片
def get_cut_img(input_dir):
    paths = glob(input_dir + '/*.*')
    for img_name in paths:
        base_name = img_name.split('/')[-1]
        print(img_name)
        img = cv2.imread(img_name)
        cropped = img[0:img.shape[0], 175:img.shape[1]]
        cv2.imwrite(os.path.join("/home/user/testImg1/result4", "{}_c.jpg".format(base_name.split('.')[0])), cropped)


if __name__ == '__main__':
    #加载模型
    model, converter = crnnSource()

    main('/home/user/testImg1/sam')
    # get_cut_img('/home/user/testImg1/4')
