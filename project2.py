import cv2
import numpy as np

WidthImg,HeightImg = 480,640

cap = cv2.VideoCapture(1)
cap.set(3, WidthImg)
cap.set(4, HeightImg)
cap.set(10,150)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def preProcess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)#膨胀，连接断开的边缘
    imgThres = cv2.erode(imgDial,kernel,iterations=1)#腐蚀，去掉细小的白点
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        #print(len(cnt))
        if area > 5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)#轮廓周长
            #print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)#拟合成规整多边形，
            #print(len(approx))#顶点数量
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
            # objCor = len(approx)
            # x, y, w, h = cv2.boundingRect(approx)  
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)

    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),dtype=np.int32)
    add = myPoints.sum(1)
    #print("add",add)

    myPointsNew[0] = myPoints[np.argmin(add)]#左上角坐标
    myPointsNew[3] = myPoints[np.argmax(add)]#右下角坐标
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]#右上角坐标
    myPointsNew[2] = myPoints[np.argmax(diff)]#左下角坐标
    #print("myPointsNew",myPointsNew)
    return myPointsNew


def getWarp(img,biggest):
    
    # print(biggest)
    # print(biggest.shape)
    # reorder(biggest)
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[WidthImg,0],[0,HeightImg],[WidthImg,HeightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(WidthImg,HeightImg))
    
    imgCropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(WidthImg,HeightImg))

    return imgCropped

while True:
    success, img = cap.read()
    img=cv2.resize(img,(WidthImg,HeightImg))
    imgContour = img.copy()
    imgThres = preProcess(img)
    biggest = getContours(imgThres)
    #print(biggest)
    if biggest.size != 0:  # 只有找到4个点才做透视！
        imgWarp = getWarp(img, biggest)
        imageArray = ([img,imgContour],
                  [imgThres,imgWarp])
    else:                  # 找不到就显示原图，不报错
        imageArray = ([img,img],
                  [imgThres,img])

    
    stackedImages = stackImages(0.6,imageArray)

    cv2.imshow("Result", stackedImages)
    #cv2.imshow("Warped", imgWarp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break