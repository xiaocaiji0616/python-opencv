import cv2
import numpy as np

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

def getContours(img):
    #cv2.RETR_EXTERNAL：只检测外轮廓
    #cv2.CHAIN_APPROX_NONE：存储所有的轮廓点，不进行压缩
    #返回值：contours是一个列表，包含了所有检测到的轮廓，里面每个元素是一个轮廓：np 数组 (N,1,2)；hierarchy是一个数组，描述了轮廓之间的层次关系
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours = [轮廓1, 轮廓2, 轮廓3, 轮廓4, 轮廓5......],轮廓 1：第 1 个物体的边缘一圈坐标
    #单个轮廓 cnt 的形状：(200, 1, 2)
    #200：这个正方形轮廓一共有 200 个点（一圈的像素点）
    #1：固定格式，没用，只是 OpenCV 规定
    #2：每个点包含 x 和 y
    #3. 每个点长这样：[[ 50 40 ]]
    # [
    # [[50, 40]],   # 第1个点 (x=50, y=40)
    # [[50, 41]],   # 第2个点
    # [[50, 42]],   # 第3个点
    # [[51, 42]],   # ...
    # ...
    # ]
    print(len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        print(len(cnt))
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)#轮廓周长
            print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)#拟合成规整多边形,返回值：approx是一个np数组(N,1,2)，包含了拟合后的多边形的顶点坐标
            print(len(approx))#顶点数量
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)  

            if objCor == 3: objectType = "Tri" 
            elif objCor == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05: objectType = "Square"
                else: objectType = "Rectangle"
            elif objCor > 4: objectType = "Circle"
            else: objectType = "None"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, objectType, 
                        (x + (w // 2) - 10, y + (h // 2) - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)

path = "Resources/shapes.png"
img = cv2.imread(path)
imgContour = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)

getContours(imgCanny)


imgBlank = np.zeros_like(img)

imgStack = stackImages(0.6,([img,imgGray,imgBlur],[imgCanny,imgContour,imgBlank]))


cv2.imshow("Stacked Images",imgStack)
cv2.waitKey(0)