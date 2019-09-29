import sys
import numpy as np

class Config:
#     MEAN=np.float32([102.9801, 115.9465, 122.7717])
#     #MEAN=np.float32([100.0, 100.0, 100.0])
#     TEST_GPU_ID=0
#     SCALE=900
#     MAX_SCALE=1500
#     TEXT_PROPOSALS_WIDTH=0
#     MIN_RATIO=0.01
#     LINE_MIN_SCORE=0.6
#     TEXT_LINE_NMS_THRESH=0.3
#     MAX_HORIZONTAL_GAP=30
#     TEXT_PROPOSALS_MIN_SCORE=0.7
#     TEXT_PROPOSALS_NMS_THRESH=0.3
#     MIN_NUM_PROPOSALS=0
#     MIN_V_OVERLAPS=0.6
#     MIN_SIZE_SIM=0.6


    MEAN=np.float32([102.9801, 115.9465, 122.7717])
    #MEAN=np.float32([100.0, 100.0, 100.0])
    TEST_GPU_ID=0
    SCALE=900                       # min{reshape[0],reshape[1]}>=SCALE ||
    MAX_SCALE=1500                  # max{reshape[0],reshape[1]}<=MAX_SCALE
    TEXT_PROPOSALS_WIDTH=16         # width of proposals
    MIN_RATIO=0.5                   # min width/height ratio（0.5:height>32)
    LINE_MIN_SCORE=0.9              #
    TEXT_LINE_NMS_THRESH=0.3        # 非极大值抑制阈值?
    MAX_HORIZONTAL_GAP=50           # 水平方向最大间隔（框与框之间的合并）        TODO:finetune   50     16
    TEXT_PROPOSALS_MIN_SCORE=0.7    # 判断一行是否为文字的评分阈值
    TEXT_PROPOSALS_NMS_THRESH=0.2
    MIN_NUM_PROPOSALS=0             # min count of proposals                TODO:finetune   0       4
    MIN_V_OVERLAPS=0.7              # 垂直方向最小交并比(用于拼接文本行)        TODO:finetune   0.7     0.7
    MIN_SIZE_SIM = 0.7             # 推荐框大小相似度最小值(用于拼接文本行)      TODO:finetune   0.7    0.83