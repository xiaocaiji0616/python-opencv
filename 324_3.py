# OpenCV 主库：负责图像读取、颜色空间转换、掩码运算、窗口显示。
import cv2 as cv
# NumPy 常用于处理图像数组（本例没有直接调用，但图像本身就是 ndarray）。
import numpy as np
# ---------------------------
# 模块整体功能
# ---------------------------
# 把 logo 图叠加到背景图的右上角：
# 1) 先从背景图裁剪出一个与 logo 同尺寸的 ROI 区域；
# 2) 再用阈值法生成掩码，把 logo 背景和前景分离；
# 3) 最后把“ROI 背景”与“logo 前景”合成后写回原图。

# cv.imread(filename, flags=cv.IMREAD_COLOR)
# 参数:
# - filename: 图像路径（字符串）。
# - flags: 读取方式，默认彩色读取。
# 返回:
# - 成功: numpy.ndarray，形状通常是 (高, 宽, 通道)，BGR 顺序。
# - 失败: None（路径错误或文件损坏时常见）。
img1 = cv.imread("Resources/grassland.png")
img2 = cv.imread("Resources/opencv-logo.png")

# ndarray.shape 属性返回图像维度。
# 对彩色图通常是 (rows, cols, channels) => (高, 宽, 通道数)。
rows1, cols1, channels1 = img1.shape
rows2, cols2, channels2 = img2.shape

# 从背景图中截取 ROI（Region of Interest，感兴趣区域）。
# 切片规则是 [起始:结束]，结束索引不包含在内。
# 这里取:
# - 行: 0 到 rows2（顶部 rows2 行）
# - 列: cols1-cols2 到 cols1（最右侧 cols2 列）
# 得到一个与 logo 同尺寸的区域，后面可直接逐像素融合。
roi = img1[0:rows2, (cols1 - cols2):cols1]

# cv.cvtColor(src, code)
# 参数:
# - src: 输入图像。
# - code: 颜色转换代码，这里 BGR -> GRAY。
# 返回:
# - 转换后的图像（这里是单通道灰度图）。
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# cv.threshold(src, thresh, maxval, type)
# 参数:
# - src: 输入灰度图。
# - thresh: 阈值（这里是 240）。
# - maxval: 超过阈值后赋的值（这里是 255）。
# - type: 阈值类型（THRESH_BINARY: 大于阈值设为 maxval，否则设为 0）。
# 返回:
# - ret: 实际使用的阈值（普通阈值时一般等于 thresh）。
# - mask: 二值图，像素只有 0 或 255。
ret, mask = cv.threshold(img2gray, 240, 255, cv.THRESH_BINARY)

# 打印 mask 的基础信息和部分数值，便于观察掩码内容。
print("mask 形状:", mask.shape)
print("mask 数据类型:", mask.dtype)
print("mask 中的唯一像素值:", np.unique(mask))
print("mask 左上角 10x10 数值:\n", mask[:10, :10])
# 如需打印全部 mask 数值，可取消下面两行注释（输出会很长）。
# np.set_printoptions(threshold=np.inf, linewidth=200)
# print("mask 全部数值:\n", mask)

# cv.bitwise_not(src)
# 参数:
# - src: 输入图像（这里是二值掩码）。
# 返回:
# - 每个像素按位取反后的图像（0 <-> 255）。
mask_inv = cv.bitwise_not(mask)

# 打印 mask_inv 的基础信息和部分数值，便于和 mask 对照观察。
print("mask_inv 形状:", mask_inv.shape)
print("mask_inv 数据类型:", mask_inv.dtype)
print("mask_inv 中的唯一像素值:", np.unique(mask_inv))
print("mask_inv 左上角 10x10 数值:\n", mask_inv[:10, :10])
# 如需打印全部 mask_inv 数值，可取消下面两行注释（输出会很长）。
# np.set_printoptions(threshold=np.inf, linewidth=200)
# print("mask_inv 全部数值:\n", mask_inv)




# cv.bitwise_and(src1, src2, mask=...)
#按位与
# 参数:
# - src1, src2: 两张同尺寸图像（这里分别传相同图像，常见写法）。
# - mask: 单通道掩码，mask 为 255 的位置才保留像素。
# 返回:
# - 按位与结果图像。
# 作用:
# - img1_bg: 从 ROI 中保留“背景可见”部分。
# - img2_fg: 从 logo 中保留“前景实体”部分。
img1_bg = cv.bitwise_and(roi, roi, mask=mask)
img2_fg = cv.bitwise_and(img2, img2, mask=mask_inv)

# cv.add(src1, src2)
# 参数:
# - src1, src2: 两张同尺寸同类型图像。
# 返回:
# - 饱和相加结果（超过 255 会截断到 255，不会像 numpy 那样溢出回绕）。
# 作用: 合成“ROI 背景 + logo 前景”。
dst = cv.add(img1_bg, img2_fg)

# 把合成结果写回原图的同一块 ROI，完成贴图。
img1[0:rows2, (cols1 - cols2):cols1] = dst

# cv.imshow(winname, mat)
# 参数:
# - winname: 窗口名（字符串）。
# - mat: 要显示的图像。
# 返回:
# - 无返回值（主要用于显示窗口）。
# 展示 logo 的灰度图，方便观察阈值分割前的输入。
cv.imshow("img2gray", img2gray)
cv.imshow("res", img1)


# cv.waitKey(delay)
# 参数:
# - delay: 等待毫秒数。0 表示一直等待，直到按键。
# 返回:
# - 按键编码（int），没按到键时会返回 -1（delay>0 时常见）。
cv.waitKey(0)

# cv.destroyAllWindows()
# 作用: 关闭当前程序创建的所有 OpenCV 窗口。
cv.destroyAllWindows()