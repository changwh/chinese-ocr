from ctpn.ctpn.model import ctpn
from ctpn.ctpn.detectors import TextDetector
from ctpn.ctpn.other import draw_boxes
import numpy as np


def text_detect(img, top=0, bottom=1, left=0, right=1):
    scores, boxes, img, f = ctpn(img, top, bottom, left, right)
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    text_recs, drawn_img = draw_boxes(img, boxes, caption='im_name', wait=True, is_display=False)
    return text_recs, drawn_img, img, f
