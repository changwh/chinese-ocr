from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QMainWindow
import sys
from UI import UI

import demo


# todo:1.运行的进度用进度条显示
# todo:2.输出重定向到textBrowser
# todo:3.输入输出路径的检查
# todo:4.对同时输入多个文件的支持


# 刷新进度条
# i = 1
# while i <= 100:
#     ui.progressBar.setValue(i)
#     time.sleep(0.1)
#     i = i + 1


class MainWindow(QMainWindow, UI.Ui_MainWindow):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_func)
        self.scene = QtWidgets.QGraphicsScene()
        self.frameNum = 0
        self.video_fps = 0
        self.video_frame_count = 0
        self.setupUi(self)

    # call Detector
    def start_detect(self):
        self.textBrowser.append("running")

        # cd = CharacterDetector()
        # self.video_fps = cd.get_video_fps(self.input_path[0][0])
        # self.video_frame_count = cd.get_video_frame_count(self.input_path[0][0])
        # cd.start_detect(self.input_path[0], self.output_path)

        demo.start(self.input_path, self.output_path)

        self.textBrowser.append("finished")

    # QGraphicsScene update
    def update_func(self):
        self.frameNum += 1

        image = QtGui.QPixmap('{}/{}_{}.jpg'.format(self.output_path,  # output path
                                                    self.input_path[0][0].split('/')[-1].split('.')[0],
                                                    self.frameNum))
        copy = image.scaledToHeight(390)  # height of graphicsView
        self.scene.addPixmap(copy)
        ui.graphicsView.setScene(self.scene)

        if self.frameNum >= int(self.video_frame_count):  # 视频总帧数
            self.pushButtonStart.setText('Start')
            self.timer.stop()
            self.frameNum = 0

    def start_stop_func(self):
        if self.pushButtonStart.text() == 'Start':
            self.pushButtonStart.setText('Stop')
            # self.timer.start(int(1000 / int(self.video_fps)))

        else:
            self.pushButtonStart.setText('Start')
            # self.timer.stop()
            self.frameNum = 0

    # start button
    @pyqtSlot()
    def on_pushButtonStart_clicked(self):
        """
        Slot documentation goes here.
        """
        self.textBrowser.append(self.lineEditOutputPath.text())

        self.start_detect()
        self.start_stop_func()

    # select input file(s)
    @pyqtSlot()
    def on_pushButtonInputPath_clicked(self):
        """
        Slot documentation goes here.
        """
        self.input_path = QtWidgets.QFileDialog.getOpenFileNames(self,
                                                            "Open file(s)",
                                                            '/',
                                                            'Video files(*.avi *.flv *.mp4);;'
                                                            'Image files(*.jpg *.jpeg *.png);;'
                                                            'All files(*.*)')
        # print(self.input_path)
        if (len(self.input_path[0]) == 1):
            self.lineEditInputPath.setText(self.input_path[0][0])
        elif (len(self.input_path[0]) > 1):
            for path in self.input_path[0]:
                self.lineEditInputPath.insert(path + ";")

    # select output folder
    @pyqtSlot()
    def on_pushButtonOutputPath_clicked(self):
        """
        Slot documentation goes here.
        """
        self.output_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                 "Select output folder",
                                                                 '/')
        # print(self.output_path)
        self.lineEditOutputPath.setText(self.output_path)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
