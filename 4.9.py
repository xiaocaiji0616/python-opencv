import cv2
import numpy as np
import cv2
import numpy as np
from skimage import util
from skimage.filters import roberts
import matplotlib.pyplot as plt

img = cv2.imread("Resources/lena.png")

img = cv2.GaussianBlur(img, (7, 7), 0)

def nothing(x):
    pass    

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

while True:
    minval = cv2.getTrackbarPos("L-H", "Trackbars"), cv2.getTrackbarPos("L-S", "Trackbars"), cv2.getTrackbarPos("L-V", "Trackbars")
    maxval = cv2.getTrackbarPos("U-H", "Trackbars"), cv2.getTrackbarPos("U-S", "Trackbars"), cv2.getTrackbarPos("U-V", "Trackbars")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, minval, maxval)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break