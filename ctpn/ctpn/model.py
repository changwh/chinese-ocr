import tensorflow as tf
from ctpn.ctpn.cfg import Config
from ctpn.ctpn.other import resize_im
from ctpn.lib.networks.factory import get_network
from ctpn.lib.fast_rcnn.config import cfg
from ctpn.lib.fast_rcnn.test import test_ctpn


def load_tf_model():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # init session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('ctpn/models/')
    saver.restore(sess, ckpt.model_checkpoint_path)
    return sess, saver, net


# init model
sess, saver, net = load_tf_model()


def ctpn(img, top, bottom, left, right):
    """
    text box detect
    """
    scale, max_scale = Config.SCALE, Config.MAX_SCALE
    resize_img, resize_ratio = resize_im(img, scale=scale, max_scale=max_scale)
    # restrict the region of interest
    roi_img = resize_img[int(resize_img.shape[0] * top): int(resize_img.shape[0] * bottom), int(resize_img.shape[1] * left): int(resize_img.shape[1] * right)]
    scores, boxes = test_ctpn(sess, net, roi_img)
    # get origin resized coordinates
    boxes[:, 1] = boxes[:, 1] + int(resize_img.shape[0] * top)
    boxes[:, 3] = boxes[:, 3] + int(resize_img.shape[0] * top)
    boxes[:, 0] = boxes[:, 0] + int(resize_img.shape[1] * left)
    boxes[:, 2] = boxes[:, 2] + int(resize_img.shape[1] * left)
    return scores, boxes, resize_img, resize_ratio
