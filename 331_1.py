import cv2
import numpy as np
def gasuss_noise(image, mean=0, var=0.01):
    #高斯噪声函数，mean：均值；var：方差
    image = np.array(image / 255, dtype=float)
    #生成高斯分布的随机数
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    
    img_noise = image + noise
    if img_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    img_noise = np.clip(img_noise, low_clip, 1.0)
    img_noise = np.uint8(img_noise * 255)
    return img_noise
def sp_noise(image, prob):
    # 椒盐噪声，image：原图像；prob：噪声比例；img_noise：加噪图像
    img_noise = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rNum = np.random.random() 
            if rNum < prob:   # 添加椒噪声
                img_noise[i][j] = 0
            elif rNum > thres:   # 添加盐噪声
                img_noise[i][j] = 255
            else:
                img_noise[i][j] = img[i][j]
    return img_noise
def random_noise(image,noise_num):
    #随机噪声，image：原图像；noise_num：添加噪音点数目
    img_noise = image
    rows, cols, chn = img_noise.shape
    # 加噪声
    for i in range(noise_num):
        #随机生成指定范围的整数
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise
img = cv2.imread('Resources/lena.png')  # 输入原图像
cv2.imshow("Origin", img)
# 添加噪声
img_gasuss = gasuss_noise(img, mean=0, var=0.01) 
img_sp_noise = sp_noise(img, 0.06) 
img_random_noise = random_noise(img,1000)
# 显示
cv2.imshow("gasuss_noise ", img_gasuss)
cv2.imshow("sp_noise", img_sp_noise)
cv2.imshow("random_noise",img_random_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()