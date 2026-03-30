import cv2 as cv  # 导入 OpenCV 库，并起别名 cv，后续所有图像/视频函数都从这里调用
import numpy as np  # 导入 NumPy 库（本示例里暂未使用，常用于图像矩阵计算）
cap = cv.VideoCapture(0)  # 打开默认摄像头设备，0 通常表示电脑的第一个摄像头
while True:  # 进入无限循环，持续读取摄像头画面
    success, img = cap.read()  # 从摄像头读取一帧图像，success 表示是否读取成功，img 是读到的帧
    cv.imshow("Video",img)  # 在名为 Video 的窗口中显示当前帧图像
    if cv.waitKey(1) & 0xFF ==ord('q'):  # 等待 1ms 获取键盘输入，按下 q 键就退出循环
        break  # 跳出 while 循环，结束视频显示
cap.release()  # 释放摄像头资源
cv.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口