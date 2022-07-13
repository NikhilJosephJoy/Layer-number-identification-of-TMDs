# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simple_TMD_calculator.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QFont
import cv2, imutils
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1150, 80)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(0, 0, 80, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(80, 0, 80, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(160, 0, 80, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(240, 0, 91, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(330, 0, 80, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(410, 0, 181, 23))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(590, 0, 80, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(670, 0, 150, 23))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(820, 0, 150, 23))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(970, 0, 80, 23))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(1050, 0, 100, 23))
        self.pushButton_10.setObjectName("pushButton_10")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 30, 851, 21))
        self.label.setFont(QFont('Arial',15))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.loadImage)
        self.pushButton_2.clicked.connect(QCoreApplication.instance().quit)
        self.pushButton_3.clicked.connect(self.red_channel)
        self.pushButton_4.clicked.connect(self.bilateral_filter)
        self.pushButton_5.clicked.connect(self.plot_inertia)
        self.lineEdit.editingFinished.connect(self.textchanged)
        self.pushButton_6.clicked.connect(self.segment_image)
        #self.pushButton.clicked.connect(self.savePhoto)
        self.pushButton_7.clicked.connect(self.sub_pixel)
        self.pushButton_8.clicked.connect(self.sam_pixel)
        self.pushButton_9.clicked.connect(self.contrast)
        self.pushButton_10.clicked.connect(self.layer_num)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.filename = None # Will hold the image address location
        self.tmp = None # Will hold the temporary image for display
        self.numclusters = None
        self.bsub = None
        self.gsub = None
        self.rsub = None
        self.bsam = None
        self.gsam = None
        self.rsam = None


    def closewindow(self):
        self.MainWindow.close()

    def layer_num(self):
        layer_num = round((self.contrast_value - 0.06)/0.062)
        if layer_num < 5:
            print("the number of layers of the selected region is : " + str(layer_num))
            self.label.setText("The number of layers of the selected region of WS2 is : " + str(layer_num))
        else:
            self.label.setText("The number of layers of the selected region is greater than five")

    def contrast(self):
        contrast_value = (self.rsub - self.rsam)/self.rsub
        print("the contrast in the red channel is : ", contrast_value)
        self.label.setText("The contrast in the red channel is : " + str(contrast_value))
        layer_num = round((contrast_value - 0.06)/0.062)
        self.contrast_value = contrast_value
        
    def sam_pixel(self,image):
        image = cv2.imread("segmented_image.png")
        def click_event(event, x, y, flags, params):
            text = ""
            font = cv2.FONT_HERSHEY_COMPLEX
            color = (255,0,0)
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x,',',y)
                text = str(x) + "," + str(y)
                color = (0,255,0)
            elif event == cv2.EVENT_RBUTTONDOWN:
                click_event.b = image[y,x,0]
                click_event.g = image[y,x,1]
                click_event.r = image[y,x,2]
                print("the blue pixel value is : ", click_event.b)
                print("the green pixel value is : ", click_event.g)
                print("the red pixel value is : ", click_event.r)
                text = str(click_event.b) + "," + str(click_event.g) + "," + str(click_event.r)
                color = (0,0,255)
            cv2.putText(image, text, (x,y), font, 0.5, color, 1, cv2.LINE_AA)
            cv2.imshow('right click on the sample region', image)
        cv2.imshow('right click on the sample region', image)
        cv2.setMouseCallback('right click on the sample region', click_event)
        cv2.waitKey()
        cv2.destroyAllWindows()
        self.rsam = click_event.r
        self.gsam = click_event.g
        self.bsam = click_event.b

    def sub_pixel(self,image):
        image = cv2.imread("segmented_image.png")
        def click_event(event, x, y, flags, params):
            text = ""
            font = cv2.FONT_HERSHEY_COMPLEX
            color = (255,0,0)
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x,',',y)
                text = str(x) + "," + str(y)
                color = (0,255,0)
            elif event == cv2.EVENT_RBUTTONDOWN:
                click_event.b = image[y,x,0]
                click_event.g = image[y,x,1]
                click_event.r = image[y,x,2]
                print("the blue pixel value is : ", click_event.b)
                print("the green pixel value is : ", click_event.g)
                print("the red pixel value is : ", click_event.r)
                text = str(click_event.b) + "," + str(click_event.g) + "," + str(click_event.r)
                color = (0,0,255)
            cv2.putText(image, text, (x,y), font, 0.5, color, 1, cv2.LINE_AA)
            cv2.imshow('right click on the substrate region', image)
        cv2.imshow('right click on the substrate region', image)
        cv2.setMouseCallback('right click on the substrate region', click_event)
        cv2.waitKey()
        cv2.destroyAllWindows()
        self.rsub = click_event.r
        self.gsub = click_event.g
        self.bsub = click_event.b
        
    def textchanged(self):
        text = self.lineEdit.text()
        print('This is the text', text)
        self.numclusters = int(text)
        print(self.numclusters)

    def loadImage(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = cv2.imread(self.filename)
        print(self.filename)
        cv2.imshow("Inspect and press enter to close", self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        #self.setPhoto(self.image)
    
    """
    def setPhoto(self,image):
         This function will take image input and resize it 
            only for display purpose and convert it to QImage
            to set at the label.
        
        self.tmp = image
        image = imutils.resize(image,width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))
    """
   
    def red_channel(self,image):
        """This function will convert the loaded image to red channel"""
        #self.tmp = image
        image = cv2.imread(self.filename)
        redchannel_image = image[:,:,2]
        red_image = np.zeros(image.shape, dtype=np.uint8)
        red_image[:,:,2] = redchannel_image
        cv2.imwrite("red_image.png", red_image)
        cv2.imshow("Image in the red channel", red_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        #frame = cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB)
        #red_image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        #self.label.setPixmap(QtGui.QPixmap.fromImage(red_image))
    
    def bilateral_filter(self,image):
        """This function will take the red channel image and filter it using Bilateral filter"""
        image = cv2.imread(self.filename)
        redchannel_image = image[:,:,2]
        red_image = np.zeros(image.shape, dtype=np.uint8)
        red_image[:,:,2] = redchannel_image
        for ii in range(1):
            bf_image = cv2.bilateralFilter(red_image,2,1,1)
        cv2.imwrite("bilateral_filtered_image.png", bf_image)
        #frame = cv2.cvtColor(bf_image, cv2.COLOR_BGR2RGB)
        #bf_image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        #self.label.setPixmap(QtGui.QPixmap.fromImage(bf_image))
        cv2.imshow("Bilateral Filtered Image", bf_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def segment_image(self,image):
        """This function will segment the image to desired number of clusters"""
        image = cv2.imread("bilateral_filtered_image.png")
        bf_image = cv2.imread("bilateral_filtered_image.png")
        bf_image = bf_image.reshape((bf_image.shape[1]*bf_image.shape[0],3))
        k = self.numclusters 
        print('This is the value of k now: ',k)

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(bf_image)

        clustered = kmeans.cluster_centers_[kmeans.labels_]
        clustered = clustered.astype('uint8')
        print('This is clustered ',clustered)
        segmented_image = clustered.reshape(image.shape[0],image.shape[1],image.shape[2])
        cv2.imwrite('segmented_image.png', segmented_image)
        #frame = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        #segmented_image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        #self.label.setPixmap(QtGui.QPixmap.fromImage(segmented_image))
        cv2.imshow("Segmented Image", segmented_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def plot_inertia(self,image):
        """This function will plot the WCSS as a function of the number of clusters. The elbow maybe chosen for segmentation"""
        bf_image = cv2.imread('bilateral_filtered_image.png')
        bf_image = bf_image.reshape((bf_image.shape[1]*bf_image.shape[0],3))
        inertia_list = []
        for i in range(1,12):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(bf_image)
            inertia = kmeans.inertia_
            inertia_list.append(inertia)
            print(inertia_list)
        fig = plt.figure(figsize=[5,7])
        ax = plt.subplot(1,1,1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.plot(list(np.arange(1,12)),inertia_list)
        ax.set_xlabel('Number of clusters, N', fontsize=16)
        ax.set_ylabel('Inertia', fontsize=16)
        ax.set_title('Elbow search for the filtered image', fontsize=18)
        ax.grid('on')
        ttl = ax.title
        ttl.set_weight('bold')
        plt.savefig('Elbow plot', dpi=600)
        plt.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "TMD Layer Number Calculator"))
        self.pushButton.setText(_translate("MainWindow", "Open"))
        self.pushButton_2.setText(_translate("MainWindow", "Close"))
        self.pushButton_3.setText(_translate("MainWindow", "Red Channel"))
        self.pushButton_4.setText(_translate("MainWindow", "Bilateral Filter"))
        self.pushButton_5.setText(_translate("MainWindow", "Plot Inertia"))
        self.lineEdit.setText(_translate("MainWindow", "Enter the number of clusters"))
        self.pushButton_6.setText(_translate("MainWindow", "Segment"))
        self.pushButton_7.setText(_translate("MainWindow", "Pick Substrate Pixel"))
        self.pushButton_8.setText(_translate("MainWindow", "Pick Sample Pixel"))
        self.pushButton_9.setText(_translate("MainWindow", "Contrast"))
        self.pushButton_10.setText(_translate("MainWindow", "Layer Number"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">TMD Layer Number Calculator</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

