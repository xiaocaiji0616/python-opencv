import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

img = cv2.imread("Resources/lena.png")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)#所有人脸的框坐标列表

for (x, y, w, h) in faces:#一张脸一组 x,y,w,h
    #cv2.rectangle(图, 左上角, 右下角, 颜色, 线宽)：在人脸上画一个蓝色矩形框
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Result",img)
cv2.waitKey(0)