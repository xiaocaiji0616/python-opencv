import cv2 as cv

# 打开摄像头
cap = cv.VideoCapture(2)

while True:
    # 读取每一帧画面
    ret, frame = cap.read()
    
    # 如果读取失败，退出循环
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # 转为灰度图（已注释，需要时取消注释）
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # 显示画面
    cv.imshow('frame', frame)
    
    # 按 q 键退出
    if cv.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv.destroyAllWindows()