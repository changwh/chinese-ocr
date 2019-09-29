from ctpn.ctpn.model import ctpn
from ctpn.ctpn.detectors import TextDetector
from ctpn.ctpn.other import draw_boxes
import numpy as np


def text_detect(img, top=0, bottom=1, left=0, right=1):
    scores, boxes, resize_img, resize_ratio = ctpn(img, top, bottom, left, right)
    text_detector = TextDetector()
    boxes = text_detector.detect(boxes, scores[:, np.newaxis], resize_img.shape[:2])
    text_recs, drawn_img = draw_boxes(resize_img, boxes, caption='im_name', wait=True, is_display=False)
    return text_recs, drawn_img, resize_img, resize_ratio
