import cv2
import numpy as np

def detect_weiqi(crop_img):
    """
    自定义函数：检测裁剪的棋子区域，判断黑白颜色，返回分类结果和二值图
    """
    # 1. 灰度化
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # 2. 二值化：用于可视化调试（不反相，避免黑白语义混淆）
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. 计算棋子中心区域的原灰度均值（比二值图更稳）
    h, w = gray.shape
    # 取中心区域（避免边缘棋盘线干扰）
    center = gray[h//4: 3*h//4, w//4: 3*w//4]
    avg_gray = np.mean(center)
    
    # 4. 分类：中心灰度高为白棋，低为黑棋
    if avg_gray > 127:
        return 'white', threshold
    else:
        return 'black', threshold

def detect_go_pieces(image_path):
    # ---------------------- 1. 图像预处理 ----------------------
    # 读取原图
    img = cv2.imread("Resources/chapter9_pics/weiqi3.png")  # 替换为你的图像路径
    if img is None:
        print("图像读取失败！")
        return
    
    # 复制原图，用于绘制结果
    result = img.copy()
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊：5×5核，降噪，避免噪声误检
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---------------------- 2. 霍夫圆变换检测棋子 ----------------------
    # 核心参数：根据棋子大小调整minRadius/maxRadius，minDist设为棋子直径避免重叠
    circles = cv2.HoughCircles(
        blur,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=40,          # 棋子中心最小距离（根据棋盘格子大小调整）
        param1=50,           # Canny高阈值
        param2=30,           # 累加器阈值
        minRadius=15,        # 棋子最小半径
        maxRadius=30         # 棋子最大半径
    )

    # 若未检测到圆，直接返回
    if circles is None:
        print("未检测到围棋棋子！")
        cv2.imshow("原图", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 转换为整数坐标（霍夫圆返回浮点数）
    circles = np.uint16(np.around(circles))

    # ---------------------- 3. 绘制圆环+分类颜色（对应右图功能） ----------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in circles[0, :]:
        # 绘制绿色外圆（对应右图的绿色圆环）
        cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 绘制红色圆心（对应右图的红色圆心）
        cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)

        # 裁剪棋子局部区域
        x, y, r = i
        # 边界处理：避免裁剪超出图像范围
        y1, y2 = max(0, y - r), min(result.shape[0], y + r)
        x1, x2 = max(0, x - r), min(result.shape[1], x + r)
        crop_img = img[y1:y2, x1:x2]

        # 检测棋子颜色
        txt, threshold = detect_weiqi(crop_img)
        print(f"棋子({x},{y}) 颜色：{'黑色' if txt == 'black' else '白色'}")

        # 在结果图上标注颜色文字（可选）
        cv2.putText(result, txt, (x-10, y-10), font, 0.5, (255, 0, 0), 2)

        # 显示中间结果（可选，调试用）
        # cv2.imshow('threshold', threshold)
        # cv2.imshow('crop_img', crop_img)
        # cv2.waitKey(100)

    # ---------------------- 4. 显示最终结果 ----------------------
    cv2.imshow("(a) 围棋原图像", img)
    cv2.imshow("(b) 检测出圆环的围棋棋子", result)
    # 移动窗口，避免重叠
    cv2.moveWindow("(b) 检测出圆环的围棋棋子", x=img.shape[1], y=0)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------- 运行检测 ----------------------
if __name__ == "__main__":
    # 替换为你的图像路径
    detect_go_pieces("Resources/chapter9_pics/weiqi3.png")  # 改为你的围棋图像路径