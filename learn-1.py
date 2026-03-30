import cv2
import numpy
from PIL import Image
import pandas as pd

im = Image.open("Resources/lena.png")  # 打开图片
im.show()  # 显示图片
im_grey = im.convert("L")
matrix = numpy.asarray(im_grey)  # 转换成矩阵
dataframe = pd.DataFrame(data=matrix)  # 转换成dataframe的格式
print(dataframe)  # 打印矩阵