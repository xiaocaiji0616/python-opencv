import cv2
import numpy as np

# ---------------------- 1) 初始种子选择 ----------------------
def originalSeed(gray, th):
    """
    根据灰度阈值从图像中提取多个种子点。
    核心目的：给每颗米粒找 1 个中心点（区域生长的起点）

    参数:
        gray: 输入灰度图, shape=(H, W), dtype=uint8。
        th: 阈值, 大于该值的像素先进入候选区域。

    返回:
        seeds: list[(x, y)]
               每个元素是一个种子坐标(行, 列), 将用于后续区域生长。

    实现思路:
        1. 对 gray 做二值化, 获得候选前景区域。
        2. 在候选区域中逐个提取连通块。
        3. 每个连通块收缩到单像素, 作为该连通块的一个代表种子。
    """
    ret, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    # thresh: 候选种子区域二值图。灰度高于阈值的位置会变亮(非零)。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 3x3 椭圆结构元, 用于形态学膨胀/腐蚀。
    thresh_copy = thresh.copy()
    # thresh_copy 用于“已处理区域”的擦除, 避免重复提取同一连通块。
    thresh_B = np.zeros(gray.shape, np.uint8)
    # thresh_B 是连通块提取过程中的临时画布。
    seeds = []  # 保存最终种子坐标
    
    # 循环直到 thresh_copy 全部为 0: 说明所有候选连通块都处理过了。
    while thresh_copy.any():
        #.any() 判断 thresh_copy 中是否还有非零元素, 即是否还有未处理的候选区域。
        Xa_copy, Ya_copy = np.where(thresh_copy > 0)
        # 取当前未处理区域中的第一个前景点, 作为本轮连通块生长起点。
        thresh_B[Xa_copy[0], Ya_copy[0]] = 255
        
        # 连通块提取: 反复膨胀 thresh_B, 再与总候选区域 thresh 取交集,
        # 得到“与起点连通”的整块区域。
        for i in range(200):
            dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
            #.dilate() 对 thresh_B 进行膨胀, 扩大前景区域.
            #thresh_B: 当前连通块的二值图; kernel: 结构元; iterations=1: 每次膨胀一次。
            thresh_B = cv2.bitwise_and(thresh, dilation_B)
            #.bitwise_and() 取 thresh 和 dilation_B 的交集, 保持在候选区域内。

        
        # 当前连通块像素坐标。将其从 thresh_copy 中清零, 标记为已处理。
        Xb, Yb = np.where(thresh_B > 0)
        thresh_copy[Xb, Yb] = 0
        
        # 将当前连通块不断腐蚀, 直到只剩一个像素。
        # 这个像素可视为该连通块的代表种子点。
        while str(thresh_B.tolist()).count("255") > 1:
            #.tolist() 将 thresh_B 转换为 Python 列表, 方便字符串操作。[[0],[255],[0]] 形式的列表中, "255" 出现的次数即为当前连通块的像素数量。
            #count("255") 统计 thresh_B 中值为 255 的像素数量, 即当前连通块的像素数。
            thresh_B = cv2.erode(thresh_B, kernel, iterations=1)  # 腐蚀操作
        
        X_seed, Y_seed = np.where(thresh_B > 0)  # 取种子坐标

        if X_seed.size > 0 and Y_seed.size > 0:
            seeds.append((X_seed[0], Y_seed[0]))  # 将种子坐标写入seeds
        
        # 清空临时画布, 准备处理下一块连通区域。
        thresh_B[Xb, Yb] = 0
    return seeds

# ---------------------- 2) 区域生长 ----------------------
def regionGrow(gray, seeds, thresh, p):
    """
    基于种子点执行区域生长分割。

    参数:

        gray: 输入灰度图, shape=(H, W)。
        seeds: 初始种子列表 list[(x, y)]。
        thresh: 生长阈值, 控制“灰度相似性”。
                阈值越小, 生长越严格; 阈值越大, 区域越容易扩张。
        p: 邻域类型, 4 或 8。

    返回:
        seedMark: 生长结果二值图, 前景为 255, 背景为 0。

    实现思路:
        从种子队列中反复取点, 检查邻域像素。
        如果邻域像素与当前点灰度差小于 thresh 且未访问,
        则将其标为前景并继续入队, 直到队列为空。
    """
    seedMark = np.zeros(gray.shape)
    if p == 8:  # 八邻域
        connection = [(-1, -1), (-1, 0), (-1, 1),
                      (0, 1),(1, 1),(1, 0),         
                      (1, -1), (0, -1)  ]
        #connection 列表定义了八邻域的相对坐标: 左上、正上、右上、正右、右下、正下、左下、正左。
    elif p == 4:  # 四邻域
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # 队列为空时停止: 说明没有新像素满足生长条件。
    while len(seeds) != 0:
        # 取出队首种子点(广度优先风格)。
        pt = seeds.pop(0)
        #pop(0) 从 seeds 列表中取出第一个元素,并删除 作为当前生长点 pt。
        for i in range(p):
            #range(p) 循环访问当前点的 p 个邻域位置。
            tmpX = pt[0] + connection[i][0]
            tmpY = pt[1] + connection[i][1]
            
            # 越界像素直接跳过。
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue
            
            # 生长准则:
            # 1) 与当前点灰度差小于阈值;
            # 2) 该点尚未被标记。
            if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                #gray[pt]：读取点的灰度值；abs() = 取绝对值；seedMark[tmpX, tmpY] == 0：确保该点未被访问过。
                seedMark[tmpX, tmpY] = 255
                seeds.append((tmpX, tmpY))
    
    return seedMark



if __name__ == '__main__':
    img = cv2.imread("resources/rice.jpeg")  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original image", gray)
    
    # 第一步: 提取初始种子点
    seeds = originalSeed(gray, th=180)
    # 第二步: 从种子出发做区域生长
    seedMark = regionGrow(gray, seeds, thresh=8, p=8)
    
    cv2.imshow("seedMark", seedMark)
    cv2.waitKey(0)
    cv2.destroyAllWindows()