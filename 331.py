import cv2
import numpy as np
import matplotlib.pyplot as plt
def log_plot(c):   #绘制曲线
    x = np.arange(0, 255, 0.01)
    y = c * np.log(1 + x)
    plt.plot(x, y, 'r', linewidth=1)
    plt.title('logarithmic')
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.show()
def log(c, img_Gray):   #对数变换
    output = c * np.log(1.0 + img_Gray)
    output = np.uint8(output + 0.5)
    return output
img = cv2.imread('Resources/lena.png')
img_Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
c=45
log_plot(c)             # 绘制对数变换曲线
result = log(c, img_Gray)  # 图像灰度对数变换

cv2.imshow("Origin", img_Gray)
cv2.imshow("Logarithmic transformation", result)
cv2.waitKey(0)
cv2.destroyAllWindows()