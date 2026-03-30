import cv2
# 1. 打开笔记本摄像头（0为默认摄像头）
videoCapture = cv2.VideoCapture(0)
# 2. 自动读取摄像头的帧率、尺寸、总帧数
fps = videoCapture.get(cv2.CAP_PROP_FPS)
width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fnums = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
if fps <= 0:
    fps = 20
# 打印读取到的参数
print(f"摄像头帧率: {fps}")
print(f"画面尺寸: {size}")
print(f"总帧数: {fnums}")
# 3. 创建视频写入器（保存到本地 output.avi）
# XVID 编码兼容性好，保存路径可自行修改
video_writer = cv2.VideoWriter(
    'output1.avi',
    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
    fps,
    size
)
# 4. 循环读取摄像头画面并保存
while True:
    # 读取一帧
    success, frame = videoCapture.read()
    if not success:
        print("读取摄像头失败")
        break
    # 显示画面
    cv2.imshow('Camera', frame)
    # 写入视频文件
    video_writer.write(frame)
    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 5. 释放资源
videoCapture.release()
video_writer.release()
cv2.destroyAllWindows()
# 6. 播放刚刚保存的视频
playback = cv2.VideoCapture('output1.avi')
if not playback.isOpened():
    print('无法打开保存的视频 output1.avi')
else:
    delay = max(1, int(1000 / fps))
    print("开始播放已保存视频，按 q 退出播放")
    while True:
        success, frame = playback.read()
        if not success:
            break
        cv2.imshow('Playback', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    playback.release()
    cv2.destroyAllWindows()