import cv2 as cv
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QGraphicsPixmapItem
import sys
import UI
import demo
import os
import shutil
import time


# TODO:每隔多少帧进行一次检测, tool bar, stop button
class MainWindow(QMainWindow, UI.Ui_MainWindow):
    is_show_proc = False
    is_pause = False
    cur_frame = 0
    cur_progress = 0
    tot_progress = 0

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.scene = QtWidgets.QGraphicsScene()
        self.setupUi(self)
        self.graphicsView.setScene(self.scene)

    # call Detector
    def start_detect(self):
        video_name = self.lineEditInputPath.text()
        output_path = self.lineEditOutputPath.text()
        video_capture = cv.VideoCapture(video_name)

        if self.LE_start.text() and self.cur_frame == 0:
            start_frame = int(self.LE_start.text())
        else:
            start_frame = self.cur_frame
        if self.LE_end.text():
            end_frame = int(self.LE_end.text())
        else:
            end_frame = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT)) - 1

        self.textBrowser.append("----------------------running----------------------")
        self.textBrowser.append("起始帧：" + str(start_frame))
        self.textBrowser.append("结束帧：" + str(end_frame))
        self.textBrowser.append("文件名：" + video_name)
        self.textBrowser.append("输出目录：" + output_path)
        if start_frame > end_frame:
            self.textBrowser.append("帧数设置错误！起始帧应小于结束帧")
            self.LE_start.setReadOnly(False)
            self.LE_end.setReadOnly(False)
            self.start_pause_switcher()
            return

        # cur_frame == 0 意味着程序刚刚开始
        if self.cur_frame == 0:
            # 如果是暂停重新开始就不用重建文件夹
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            self.tot_progress = end_frame - start_frame + 1
            self.textBrowser.clear()

        for curent_frame in range(start_frame, end_frame + 1):
            # 若已执行到最后一帧，则使帧数输入框不为只读
            if curent_frame == end_frame:
                self.LE_start.setReadOnly(False)
                self.LE_end.setReadOnly(False)

            # 暂停时记录当前帧数
            if self.is_pause:
                self.cur_frame = curent_frame
                return

            t = time.time()
            # 获得log信息，图片
            result, frame, str_ui = demo.start_video_byframe(
                video_name, output_path, video_capture, curent_frame, self.is_show_proc)

            self.textBrowser.append("----------------------" + video_name + ":" + str(curent_frame) + "----------------------")

            if self.is_show_proc:
                # 显示过程
                self.textBrowser.append(str_ui)

            self.textBrowser.append("Frame number:{}, It takes time:{}s".format(curent_frame, time.time() - t))
            self.textBrowser.append("识别结果:")
            for key in result:
                self.textBrowser.append(result[key][1])

            # 将得到的ndarray转换为pixmap
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            q_img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            item = QGraphicsPixmapItem(pixmap)

            # 自适应图片尺寸
            self.graphicsView.fitInView(item, mode=Qt.KeepAspectRatio)

            # 清除之前的图片，加入新的图片
            self.scene.clear()
            self.scene.addItem(item)

            # 更新进度条
            self.cur_progress += 1
            ui.progressBar.setValue((self.cur_progress / self.tot_progress) * 100)

            # 实时刷新ui
            app.processEvents()
        else:
            self.cur_progress = 0
            self.cur_frame = 0
            self.tot_progress = 0
            self.textBrowser.append("输出路径:" + self.output_path)
            self.textBrowser.append("----------------------finished----------------------")
            self.start_pause_switcher()

    def start_pause_switcher(self):
        # 控制开始停止
        if self.pushButtonStart.text() == 'Start':
            self.pushButtonStart.setText('Pause')
            self.start_detect()
            self.is_pause = False
        else:
            self.pushButtonStart.setText('Start')
            self.is_pause = True

    # CheckBox
    @pyqtSlot()
    def on_CheckBoxIsShowProc_clicked(self):
        if self.is_show_proc:
            self.is_show_proc = False
        else:
            self.is_show_proc = True

    # start button
    @pyqtSlot()
    def on_pushButtonStart_clicked(self):
        # 输入验证
        if not self.lineEditInputPath.text():
            self.textBrowser.append("请选择输入文件！")
            return
        elif not self.lineEditOutputPath.text():
            self.textBrowser.append("请选择输出目录！")
            return

        # 设置开始帧和结束帧输入框为只读
        self.LE_start.setReadOnly(True)
        self.LE_end.setReadOnly(True)

        # 开始暂停显示切换，按键显示为start时点击调用检测程序
        self.start_pause_switcher()

    # select input file
    @pyqtSlot()
    def on_pushButtonInputPath_clicked(self):
        self.input_path = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                "Open file",
                                                                '/',
                                                                'Video files(*.avi *.flv *.mp4);;'
                                                                'All files(*.*)')
        if self.input_path[0]:
            self.lineEditInputPath.setText(self.input_path[0])

    # select output folder
    @pyqtSlot()
    def on_pushButtonOutputPath_clicked(self):
        self.output_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                      "Select output folder",
                                                                      '/')
        self.lineEditOutputPath.setText(self.output_path)

    # exit
    @pyqtSlot()
    def on_pushButtonExit_clicked(self):
        sys.exit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
