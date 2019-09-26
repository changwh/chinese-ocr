from ctpn.ctpn.model import ctpn
from ctpn.ctpn.detectors import TextDetector
from ctpn.ctpn.other import draw_boxes
import numpy as np
import cv2


def draw_text_proposals(img, text_proposals):

    tmp = img.copy()
    color = tuple([0, 255, 0])
    line_size = 3

    for text_proposal in text_proposals:
        x1 = text_proposal[0]
        x2 = text_proposal[2]
        x3 = text_proposal[0]
        x4 = text_proposal[2]
        y1 = text_proposal[1]
        y2 = text_proposal[1]
        y3 = text_proposal[3]
        y4 = text_proposal[3]

        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), color, line_size)
        cv2.line(tmp, (int(x1), int(y1)), (int(x3), int(y3)), color, line_size)
        cv2.line(tmp, (int(x4), int(y4)), (int(x2), int(y2)), color, line_size)
        cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), color, line_size)

    return tmp


def text_detect(img, top=0, bottom=1, left=0, right=1):
    scores, boxes, resize_img, resize_ratio = ctpn(img, top, bottom, left, right)
    text_detector = TextDetector()
    boxes, _ = text_detector.detect(boxes, scores[:, np.newaxis], resize_img.shape[:2])
    text_recs, drawn_img = draw_boxes(resize_img, boxes, caption='im_name', wait=True, is_display=False)
    return text_recs, drawn_img, resize_img, resize_ratio


# TODO;修改
def test_text_detect(img, top=0, bottom=1, left=0, right=1):
    height = img.shape[0]
    width = img.shape[1]

    scores, boxes, img, f = ctpn(img, top, bottom, left, right)
    textdetector = TextDetector()
    boxes, text_proposals = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    drawn_proposals_img = draw_text_proposals(img, text_proposals)
    text_recs, drawn_img = draw_boxes(drawn_proposals_img, boxes, caption='im_name', wait=True, is_display=False)
    drawn_img = cv2.resize(drawn_img, (width, height))
    return text_recs, drawn_img, img, f
