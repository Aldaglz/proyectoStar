# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1511, 797)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Top_frame = QtWidgets.QFrame(self.centralwidget)
        self.Top_frame.setMinimumSize(QtCore.QSize(0, 40))
        self.Top_frame.setMaximumSize(QtCore.QSize(16777215, 40))
        self.Top_frame.setStyleSheet("background-color: rgb(38, 255, 160);\n"
"")
        self.Top_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Top_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Top_frame.setObjectName("Top_frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.Top_frame)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_menu = QtWidgets.QPushButton(self.Top_frame)
        self.btn_menu.setMinimumSize(QtCore.QSize(200, 35))
        self.btn_menu.setMaximumSize(QtCore.QSize(16777215, 35))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.btn_menu.setFont(font)
        self.btn_menu.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}\n"
"")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../DinamicMenu/icons/menu.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_menu.setIcon(icon)
        self.btn_menu.setIconSize(QtCore.QSize(32, 32))
        self.btn_menu.setObjectName("btn_menu")
        self.horizontalLayout_2.addWidget(self.btn_menu)
        spacerItem = QtWidgets.QSpacerItem(451, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.btn_minimizar = QtWidgets.QPushButton(self.Top_frame)
        self.btn_minimizar.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}")
        self.btn_minimizar.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../DinamicMenu/icons/minus.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_minimizar.setIcon(icon1)
        self.btn_minimizar.setIconSize(QtCore.QSize(32, 32))
        self.btn_minimizar.setObjectName("btn_minimizar")
        self.horizontalLayout_2.addWidget(self.btn_minimizar)
        self.btn_restaurar = QtWidgets.QPushButton(self.Top_frame)
        self.btn_restaurar.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}")
        self.btn_restaurar.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../DinamicMenu/icons/minimize-2.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_restaurar.setIcon(icon2)
        self.btn_restaurar.setIconSize(QtCore.QSize(32, 32))
        self.btn_restaurar.setObjectName("btn_restaurar")
        self.horizontalLayout_2.addWidget(self.btn_restaurar)
        self.btn_maximizar = QtWidgets.QPushButton(self.Top_frame)
        self.btn_maximizar.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}")
        self.btn_maximizar.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../DinamicMenu/icons/maximize-2.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_maximizar.setIcon(icon3)
        self.btn_maximizar.setIconSize(QtCore.QSize(32, 32))
        self.btn_maximizar.setObjectName("btn_maximizar")
        self.horizontalLayout_2.addWidget(self.btn_maximizar)
        self.btn_cerrar = QtWidgets.QPushButton(self.Top_frame)
        self.btn_cerrar.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}\n"
"")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("../DinamicMenu/icons/x-octagon.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_cerrar.setIcon(icon4)
        self.btn_cerrar.setIconSize(QtCore.QSize(32, 32))
        self.btn_cerrar.setObjectName("btn_cerrar")
        self.horizontalLayout_2.addWidget(self.btn_cerrar)
        self.verticalLayout.addWidget(self.Top_frame)
        self.Bottom_frame = QtWidgets.QFrame(self.centralwidget)
        self.Bottom_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Bottom_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Bottom_frame.setObjectName("Bottom_frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.Bottom_frame)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Lateral_frame = QtWidgets.QFrame(self.Bottom_frame)
        self.Lateral_frame.setEnabled(True)
        self.Lateral_frame.setMinimumSize(QtCore.QSize(0, 0))
        self.Lateral_frame.setMaximumSize(QtCore.QSize(200, 16777215))
        self.Lateral_frame.setAutoFillBackground(False)
        self.Lateral_frame.setStyleSheet("QFrame{\n"
"background-color: rgb(38, 255, 208);\n"
"}\n"
"\n"
"QPushButton{\n"
"background-color: #7de5ff;\n"
"border-top-left-radius:20px;\n"
"border-bottom-left-radius:20px;\n"
"\n"
"font: 75 20pt \"Arial Narrow\";\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"\n"
"backgroud-color:white;\n"
"border-top.left.radius: 20px;\n"
"border-bottom-left-radius:20px;\n"
"font: 75 12pt \"Arial Narrow\";\n"
"\n"
"}")
        self.Lateral_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Lateral_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Lateral_frame.setObjectName("Lateral_frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.Lateral_frame)
        self.verticalLayout_2.setContentsMargins(0, 5, 0, 0)
        self.verticalLayout_2.setSpacing(5)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btn_Home = QtWidgets.QPushButton(self.Lateral_frame)
        self.btn_Home.setMinimumSize(QtCore.QSize(0, 40))
        self.btn_Home.setMaximumSize(QtCore.QSize(16777215, 40))
        self.btn_Home.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}\n"
"\n"
"")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icons/home.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_Home.setIcon(icon5)
        self.btn_Home.setIconSize(QtCore.QSize(24, 24))
        self.btn_Home.setObjectName("btn_Home")
        self.verticalLayout_2.addWidget(self.btn_Home)
        self.btn_Streaming = QtWidgets.QPushButton(self.Lateral_frame)
        self.btn_Streaming.setMinimumSize(QtCore.QSize(0, 40))
        self.btn_Streaming.setMaximumSize(QtCore.QSize(16777215, 40))
        self.btn_Streaming.setStyleSheet("QPushButton{\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}\n"
"")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("../DinamicMenu/icons/eye.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_Streaming.setIcon(icon6)
        self.btn_Streaming.setIconSize(QtCore.QSize(24, 24))
        self.btn_Streaming.setObjectName("btn_Streaming")
        self.verticalLayout_2.addWidget(self.btn_Streaming)
        spacerItem1 = QtWidgets.QSpacerItem(20, 250, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.ad = QtWidgets.QLabel(self.Lateral_frame)
        self.ad.setMinimumSize(QtCore.QSize(181, 141))
        self.ad.setMaximumSize(QtCore.QSize(181, 141))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(22)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.ad.setFont(font)
        self.ad.setStyleSheet("")
        self.ad.setText("")
        self.ad.setTextFormat(QtCore.Qt.RichText)
        self.ad.setPixmap(QtGui.QPixmap("logonew.png"))
        self.ad.setScaledContents(True)
        self.ad.setObjectName("ad")
        self.verticalLayout_2.addWidget(self.ad)
        self.horizontalLayout.addWidget(self.Lateral_frame)
        self.Docker_frame = QtWidgets.QFrame(self.Bottom_frame)
        self.Docker_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Docker_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Docker_frame.setObjectName("Docker_frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.Docker_frame)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.stackedWidget = QtWidgets.QStackedWidget(self.Docker_frame)
        self.stackedWidget.setObjectName("stackedWidget")
        self.Home_page = QtWidgets.QWidget()
        self.Home_page.setObjectName("Home_page")
        self.gridLayout = QtWidgets.QGridLayout(self.Home_page)
        self.gridLayout.setObjectName("gridLayout")
        self.bg = QtWidgets.QLabel(self.Home_page)
        self.bg.setText("")
        self.bg.setPixmap(QtGui.QPixmap("icons/logoBgWhite.png"))
        self.bg.setScaledContents(True)
        self.bg.setObjectName("bg")
        self.gridLayout.addWidget(self.bg, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.Home_page)
        self.Streaming_page = QtWidgets.QWidget()
        self.Streaming_page.setObjectName("Streaming_page")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.Streaming_page)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.gb_Docker = QtWidgets.QGroupBox(self.Streaming_page)
        self.gb_Docker.setTitle("")
        self.gb_Docker.setObjectName("gb_Docker")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gb_Docker)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gb_Skeleton = QtWidgets.QGroupBox(self.gb_Docker)
        self.gb_Skeleton.setMinimumSize(QtCore.QSize(249, 281))
        self.gb_Skeleton.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.gb_Skeleton.setFont(font)
        self.gb_Skeleton.setObjectName("gb_Skeleton")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gb_Skeleton)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.lb_Skeleton = QtWidgets.QLabel(self.gb_Skeleton)
        self.lb_Skeleton.setText("")
        self.lb_Skeleton.setPixmap(QtGui.QPixmap("icons/aviso.png"))
        self.lb_Skeleton.setScaledContents(True)
        self.lb_Skeleton.setObjectName("lb_Skeleton")
        self.gridLayout_3.addWidget(self.lb_Skeleton, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.gb_Skeleton, 0, 0, 1, 1)
        self.gb_Exercise = QtWidgets.QGroupBox(self.gb_Docker)
        self.gb_Exercise.setMinimumSize(QtCore.QSize(249, 321))
        self.gb_Exercise.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.gb_Exercise.setFont(font)
        self.gb_Exercise.setFlat(True)
        self.gb_Exercise.setObjectName("gb_Exercise")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.gb_Exercise)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.lb_Exercise = QtWidgets.QLabel(self.gb_Exercise)
        font = QtGui.QFont()
        font.setFamily("Cascadia Code SemiBold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_Exercise.setFont(font)
        self.lb_Exercise.setStyleSheet("background-color: rgb(239, 220, 175);")
        self.lb_Exercise.setTextFormat(QtCore.Qt.MarkdownText)
        self.lb_Exercise.setWordWrap(True)
        self.lb_Exercise.setObjectName("lb_Exercise")
        self.gridLayout_5.addWidget(self.lb_Exercise, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.gb_Exercise, 0, 2, 2, 1)
        self.gb_Angles = QtWidgets.QGroupBox(self.gb_Docker)
        self.gb_Angles.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.gb_Angles.setFont(font)
        self.gb_Angles.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.gb_Angles.setObjectName("gb_Angles")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.gb_Angles)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_3 = QtWidgets.QLabel(self.gb_Angles)
        self.label_3.setMinimumSize(QtCore.QSize(117, 21))
        self.label_3.setMaximumSize(QtCore.QSize(117, 21))
        self.label_3.setObjectName("label_3")
        self.gridLayout_8.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gb_Angles)
        self.label_7.setMinimumSize(QtCore.QSize(109, 16))
        self.label_7.setMaximumSize(QtCore.QSize(109, 16))
        self.label_7.setObjectName("label_7")
        self.gridLayout_8.addWidget(self.label_7, 0, 1, 1, 1)
        self.lb_RAA = QtWidgets.QLabel(self.gb_Angles)
        self.lb_RAA.setMinimumSize(QtCore.QSize(47, 13))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_RAA.setFont(font)
        self.lb_RAA.setObjectName("lb_RAA")
        self.gridLayout_8.addWidget(self.lb_RAA, 1, 0, 1, 1)
        self.lb_LAA = QtWidgets.QLabel(self.gb_Angles)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_LAA.setFont(font)
        self.lb_LAA.setObjectName("lb_LAA")
        self.gridLayout_8.addWidget(self.lb_LAA, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gb_Angles)
        self.label_4.setMinimumSize(QtCore.QSize(147, 16))
        self.label_4.setMaximumSize(QtCore.QSize(147, 16))
        self.label_4.setObjectName("label_4")
        self.gridLayout_8.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gb_Angles)
        self.label_6.setMinimumSize(QtCore.QSize(139, 16))
        self.label_6.setMaximumSize(QtCore.QSize(139, 16))
        self.label_6.setObjectName("label_6")
        self.gridLayout_8.addWidget(self.label_6, 2, 1, 1, 1)
        self.lb_RSA = QtWidgets.QLabel(self.gb_Angles)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_RSA.setFont(font)
        self.lb_RSA.setObjectName("lb_RSA")
        self.gridLayout_8.addWidget(self.lb_RSA, 3, 0, 1, 1)
        self.lb_LSA = QtWidgets.QLabel(self.gb_Angles)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lb_LSA.setFont(font)
        self.lb_LSA.setObjectName("lb_LSA")
        self.gridLayout_8.addWidget(self.lb_LSA, 3, 1, 1, 1)
        self.gridLayout_2.addWidget(self.gb_Angles, 2, 0, 1, 2)
        self.gb_Emotion = QtWidgets.QGroupBox(self.gb_Docker)
        self.gb_Emotion.setMinimumSize(QtCore.QSize(248, 281))
        self.gb_Emotion.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.gb_Emotion.setFont(font)
        self.gb_Emotion.setObjectName("gb_Emotion")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gb_Emotion)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lb_Emotion = QtWidgets.QLabel(self.gb_Emotion)
        self.lb_Emotion.setText("")
        self.lb_Emotion.setPixmap(QtGui.QPixmap("icons/aviso.png"))
        self.lb_Emotion.setScaledContents(True)
        self.lb_Emotion.setObjectName("lb_Emotion")
        self.gridLayout_4.addWidget(self.lb_Emotion, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.gb_Emotion, 0, 1, 1, 1)
        self.gb_Linke = QtWidgets.QGroupBox(self.gb_Docker)
        self.gb_Linke.setMaximumSize(QtCore.QSize(249, 163))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.gb_Linke.setFont(font)
        self.gb_Linke.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.gb_Linke.setTitle("")
        self.gb_Linke.setFlat(True)
        self.gb_Linke.setObjectName("gb_Linke")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.gb_Linke)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_9 = QtWidgets.QLabel(self.gb_Linke)
        self.label_9.setMinimumSize(QtCore.QSize(61, 101))
        self.label_9.setMaximumSize(QtCore.QSize(61, 101))
        self.label_9.setText("")
        self.label_9.setPixmap(QtGui.QPixmap("blutwo.png"))
        self.label_9.setScaledContents(True)
        self.label_9.setObjectName("label_9")
        self.gridLayout_9.addWidget(self.label_9, 0, 0, 1, 1)
        self.lb_vin = QtWidgets.QLabel(self.gb_Linke)
        self.lb_vin.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_vin.setObjectName("lb_vin")
        self.gridLayout_9.addWidget(self.lb_vin, 0, 1, 1, 1)
        self.gridLayout_2.addWidget(self.gb_Linke, 2, 2, 1, 1)
        self.gb_States = QtWidgets.QGroupBox(self.gb_Docker)
        self.gb_States.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.gb_States.setTitle("")
        self.gb_States.setObjectName("gb_States")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.gb_States)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.btn_TurnOn = QtWidgets.QPushButton(self.gb_States)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_TurnOn.setFont(font)
        self.btn_TurnOn.setStyleSheet("\n"
"QPushButton{\n"
"background-color: rgb(170, 255, 127);\n"
"\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}\n"
"\n"
"")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icons/video.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_TurnOn.setIcon(icon7)
        self.btn_TurnOn.setIconSize(QtCore.QSize(24, 24))
        self.btn_TurnOn.setObjectName("btn_TurnOn")
        self.gridLayout_7.addWidget(self.btn_TurnOn, 0, 0, 1, 1)
        self.btn_TurnOff = QtWidgets.QPushButton(self.gb_States)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_TurnOff.setFont(font)
        self.btn_TurnOff.setStyleSheet("\n"
"\n"
"QPushButton{\n"
"background-color: rgb(255, 85, 0);\n"
"\n"
"border:0px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"border:5px solid#aa00ff;\n"
"background-color:#ffff00;\n"
"}\n"
"\n"
"")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("icons/video-off.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_TurnOff.setIcon(icon8)
        self.btn_TurnOff.setIconSize(QtCore.QSize(23, 24))
        self.btn_TurnOff.setObjectName("btn_TurnOff")
        self.gridLayout_7.addWidget(self.btn_TurnOff, 0, 1, 1, 1)
        self.gridLayout_2.addWidget(self.gb_States, 1, 0, 1, 2)
        self.horizontalLayout_4.addWidget(self.gb_Docker)
        self.stackedWidget.addWidget(self.Streaming_page)
        self.emotions_Page = QtWidgets.QWidget()
        self.emotions_Page.setObjectName("emotions_Page")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.emotions_Page)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gb_dataEmotio = QtWidgets.QGroupBox(self.emotions_Page)
        self.gb_dataEmotio.setTitle("")
        self.gb_dataEmotio.setObjectName("gb_dataEmotio")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.gb_dataEmotio)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.lb_Cat = QtWidgets.QLabel(self.gb_dataEmotio)
        self.lb_Cat.setMinimumSize(QtCore.QSize(512, 512))
        self.lb_Cat.setMaximumSize(QtCore.QSize(512, 512))
        self.lb_Cat.setStyleSheet("background-color: rgb(255, 110, 14);\n"
"background-color: rgb(255, 255, 255);")
        self.lb_Cat.setText("")
        self.lb_Cat.setObjectName("lb_Cat")
        self.gridLayout_10.addWidget(self.lb_Cat, 0, 0, 1, 1)
        self.gridLayout_6.addWidget(self.gb_dataEmotio, 0, 0, 1, 1)
        self.stackedWidget.addWidget(self.emotions_Page)
        self.horizontalLayout_3.addWidget(self.stackedWidget)
        self.horizontalLayout.addWidget(self.Docker_frame)
        self.verticalLayout.addWidget(self.Bottom_frame)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_menu.setText(_translate("MainWindow", " MENU"))
        self.btn_Home.setText(_translate("MainWindow", "Home"))
        self.btn_Streaming.setText(_translate("MainWindow", "Streaming"))
        self.gb_Skeleton.setTitle(_translate("MainWindow", "Skeleton"))
        self.gb_Exercise.setTitle(_translate("MainWindow", "Exercise"))
        self.lb_Exercise.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\"><span style=\" font-size:14pt; font-weight:600;\">Seleccione un ejercicio desde la aplicación movil y envielo</span></p></body></html>"))
        self.gb_Angles.setTitle(_translate("MainWindow", "Arm\'s Angles"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; color:#c7211b;\">Right Arm\'s Angle</span></p><p><br/></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; color:#c7211b;\">Left Arm\'s Angle</span></p><p><br/></p></body></html>"))
        self.lb_RAA.setText(_translate("MainWindow", "0°"))
        self.lb_LAA.setText(_translate("MainWindow", "0°"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; color:#c7211b;\">Right Shoulder\'s Angle</span></p><p><br/></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; color:#c7211b;\">Left Shoulder\'s Angle</span></p><p><br/></p></body></html>"))
        self.lb_RSA.setText(_translate("MainWindow", "0°"))
        self.lb_LSA.setText(_translate("MainWindow", "0°"))
        self.gb_Emotion.setTitle(_translate("MainWindow", "Emotion"))
        self.lb_vin.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#ffffff;\">No Vinculado</span></p></body></html>"))
        self.btn_TurnOn.setText(_translate("MainWindow", "    Encender Video"))
        self.btn_TurnOff.setText(_translate("MainWindow", "    Apagar Video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())