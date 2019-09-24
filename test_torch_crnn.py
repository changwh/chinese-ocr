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
    path = './samples/crnn_best.pth'
    # path = 'crnn/samples/model_acc97.pth'   # vscode
    model.eval()
    model.load_state_dict(torch.load(path))

    # # 冻结某些参数
    # for i,para in enumerate(model.parameters()):
    #     if i < 20:
    #         para.requires_grad = False
    #     else:
    #         para.requires_grad = True

    # # 打印模型结构
    # import pprint
    # import torchsummaryX
    #
    # params = list(model.named_parameters())
    # pprint.pprint(params)
    # torchsummaryX.summary(model, torch.zeros((1,1,32,256)).cuda())

    return model, converter


def crnnOcr(image):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im
    @@text_recs:text box

    """

    # 按训练集的缩放比例缩放（假设训练图片宽度为280，训练时缩放到256），只缩放宽度(height变化的情况下效果不好)
    # w = int(image.size[0] / (280 * 1.0 / 256))
    # 按比例缩放（训练时尽量不缩放）（论文中的方法，但前提是测试样本长度大于训练样本）
    w = int(image.size[0] / (image.size[1] * 1.0 / 32))
    # 先将测试图片按比例缩放至高度为32，再将缩放后图片按训练集的比例缩放
    # w = int((image.size[0] / (image.size[1] * 1.0 / 32)) / (250 * 1.0 / 256))

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
        # gray_im = ImageOps.invert(img)
        # result2 = get_crnn_result(gray_im)

        print("Frame number:{}, It takes time:{}s".format(0, time.time() - t))
        print("---------------------------------------")
        print("识别结果:")
        print("origin:", result1)
        # print("invert:", result2)

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

    main('/home/user/testImg1/sam_font')
    # get_cut_img('/home/user/testImg1/4')
