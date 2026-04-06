import cv2


frameWidth = 640
frameHeight = 480
nPlateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
minArea = 500
color = (255,0,255)
count = 0

cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    numberPlayes = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)#所有人脸的框坐标列表

    for (x, y, w, h) in numberPlayes:#一张脸一组 x,y,w,h
        #cv2.rectangle(图, 左上角, 右下角, 颜色, 线宽)：在人脸上画一个蓝色矩形框
        area = w*h
        if area > minArea:#过滤掉面积过小的框
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(img,"Number Plate",(x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX,1,color,2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)



    cv2.imshow("Result", img)


    if cv2.waitKey(1) & 0xFF == ord('s'):

        cv2.imwrite("Resources/Scanned/NoPlate_"+str(count)+".jpg",imgRoi)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(150,265),cv2.FONT_HERSHEY_COMPLEX,
                    2,(0,0,255),2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    