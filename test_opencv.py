# 尝试导入OpenCV库
import cv2

# 打印OpenCV版本号（核心验证）
print("OpenCV安装成功！版本号：", cv2.__version__)

# 可选：打印一些基础功能，确认库完整
print("OpenCV主要模块：", dir(cv2)[:10])  # 打印前10个模块名，仅作参考