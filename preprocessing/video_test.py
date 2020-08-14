import cv2
from collections import OrderedDict
from PIL import Image

if __name__ == '__main__':
    capture = cv2.VideoCapture('F:\\Celeb-DF-v2\\Celeb-real\\id27_0005.mp4')
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = OrderedDict()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize(size=[s // 2 for s in frame.size])
        frames[i] = frame

    print(len(frames))
