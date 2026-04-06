import cv2
import numpy as np
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)


myColors = [[57,90,61,88,255,255],#绿色范围
            [23,164,172,34,255,255],#黄色范围
            [91,153,49,156,255,255]]#蓝色范围

myColorValues = [[0,255,0],#BGR格式的绿色
                 [0,255,255],#黄色
                 [255,0,0]]#蓝色

myPoints = []#[[x,y,colorId]]

def findColor(img,myColors,myColorValues):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV,lower,upper)
        x,y=getContours(mask)#循环到别的颜色时x，y=0，画圈画在左上角
        cv2.circle(imgResult,(x,y),10,myColorValues[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count += 1
       # cv2.imshow(str(color[0]),mask)
    return newPoints

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
  #  print(len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
  #      print(area)
   #     print(len(cnt))
        if area > 500:
            #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)#轮廓周长
         #   print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)#拟合成规整多边形
        #    print(len(approx))#顶点数量
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y+10        

def drawOnCanvas(myPoints,myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img,myColors,myColorValues)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        drawOnCanvas(myPoints,myColorValues)
    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break