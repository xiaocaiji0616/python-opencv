import cv2 as cv
import numpy as np

# 1. 打开默认摄像头（0 通常是笔记本内置摄像头）
cap = cv.VideoCapture(0)

# 2. 读取摄像头参数
fps = cap.get(cv.CAP_PROP_FPS)
size = (
    int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
)
fnums = cap.get(cv.CAP_PROP_FRAME_COUNT)

# 某些摄像头读取到的 fps 可能是 0，给一个默认值防止后续除零
if fps <= 0:
    fps = 20

print("FPS:", fps)
print("Size:", size)
print("Frame Count:", fnums)

# 3. 创建视频写入器，把画面保存到 output.avi
video_writer = cv.VideoWriter(
    "output.avi",
    cv.VideoWriter_fourcc(*"XVID"),
    fps,
    size,
)

# 4. 实时显示摄像头画面，同时写入视频文件
while True:
    success, img = cap.read()
    if not success:
        print("读取摄像头失败")
        break

    cv.imshow("Video", img)
    video_writer.write(img)

    # 按 q 结束录制
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# 5. 录制结束，释放资源
video_writer.release()
cap.release()
cv.destroyAllWindows()

# 6. 回放刚保存的视频
playback = cv.VideoCapture('output.avi')
if not playback.isOpened():
    print('无法打开保存的视频 output.avi')
else:
    # 按录制时的 fps 播放，让速度更接近原视频
    delay = max(1, int(1000 / fps))
    print("开始播放已保存视频，按 q 退出播放")
    while True:
        success, frame = playback.read()
        if not success:
            break

        cv.imshow('Playback', frame)
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break

    playback.release()
    cv.destroyAllWindows()
