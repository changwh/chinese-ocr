import tensorflow as tf
from ctpn.ctpn.cfg import Config
from ctpn.ctpn.other import resize_im
from ctpn.lib.networks.factory import get_network
from ctpn.lib.fast_rcnn.config import cfg
from ctpn.lib.fast_rcnn.test import test_ctpn


def load_tf_model():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
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


def ctpn(img):
    """
    text box detect
    """
    scale, max_scale = Config.SCALE, Config.MAX_SCALE
    img, f = resize_im(img, scale=scale, max_scale=max_scale)
    # restrict the region of interest
    roi_img = img[int(img.shape[0] * 0.7):img.shape[0], 0:img.shape[1]]
    scores, boxes = test_ctpn(sess, net, roi_img)
    # get origin resized coordinates
    boxes[:, 1] = boxes[:, 1] + int(img.shape[0] * 0.7)
    boxes[:, 3] = boxes[:, 3] + int(img.shape[0] * 0.7)
    return scores, boxes, img, f
