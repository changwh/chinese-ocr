# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/user/PycharmProjects/text-detection-ctpn/UI/UI.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalWidget.setGeometry(QtCore.QRect(10, 0, 780, 555))
        self.verticalWidget.setObjectName("verticalWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.graphicsView = QtWidgets.QGraphicsView(self.verticalWidget)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setMinimumSize(QtCore.QSize(640, 352))
        self.graphicsView.setMouseTracking(False)
        self.graphicsView.setAlignment(QtCore.Qt.AlignCenter)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.widget = QtWidgets.QWidget(self.verticalWidget)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.pushButtonOutput = QtWidgets.QPushButton(self.widget)
        self.pushButtonOutput.setObjectName("pushButtonOutput")
        self.horizontalLayout.addWidget(self.pushButtonOutput, 0, QtCore.Qt.AlignRight)
        self.verticalLayout.addWidget(self.widget)
        self.textBrowser = QtWidgets.QTextBrowser(self.verticalWidget)
        self.textBrowser.setEnabled(True)
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 100))
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.progressBar = QtWidgets.QProgressBar(self.verticalWidget)
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 30))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.pushButtonExit = QtWidgets.QPushButton(self.verticalWidget)
        self.pushButtonExit.setMaximumSize(QtCore.QSize(80, 30))
        self.pushButtonExit.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.pushButtonExit.setAutoFillBackground(True)
        self.pushButtonExit.setObjectName("pushButtonExit")
        self.verticalLayout.addWidget(self.pushButtonExit)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_files = QtWidgets.QAction(MainWindow)
        self.actionOpen_files.setObjectName("actionOpen_files")
        self.actionOpen_folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_folder.setObjectName("actionOpen_folder")
        self.actionAbout_Subtitle_Detector = QtWidgets.QAction(MainWindow)
        self.actionAbout_Subtitle_Detector.setObjectName("actionAbout_Subtitle_Detector")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_files)
        self.menuFile.addAction(self.actionOpen_folder)
        self.menuHelp.addAction(self.actionAbout_Subtitle_Detector)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButtonExit.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Subtitle Detector"))
        self.pushButtonOutput.setText(_translate("MainWindow", "Output"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">hello</p></body></html>"))
        self.pushButtonExit.setText(_translate("MainWindow", "Exit"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen_files.setText(_translate("MainWindow", "Open files"))
        self.actionOpen_folder.setText(_translate("MainWindow", "Open folder"))
        self.actionAbout_Subtitle_Detector.setText(_translate("MainWindow", "About Subtitle Detector"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

