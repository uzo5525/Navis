import queue
import threading

import cv2
import platform
from ultralytics import YOLO
import numpy as np

import pyttsx3
import pythoncom
import time

class Speaker:
    def __init__(self, lang='ko', rate='175'):
        self.q = queue.Queue(maxsize=1)
        self.lang = lang
        self.rate = int(rate)
        self.engine = None
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def say(self, text):
        try:
            if self.q.full():
                _ = self.q.get_nowait()
            self.q.put_nowait(text)
            return True
        except queue.Full:
            return False

    def _loop(self):
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty('rate', int(self.rate))
        for v in engine.getProperty("voices"):
            if "ko" in v.id.lower() or "korean" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        while True:
            text = self.q.get()
            try:
                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception:
                pass

os = platform.system().lower()
if os == 'windows':
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
elif os == 'linux':
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
else:
    raise TypeError('os를 인식할 수 없습니다')

model = YOLO("D:/dev/deocent/dev/models/best.pt")
model.model.names = {
    0: 'crosswalk',
    1: 'trafficlight_red',
    2: 'trafficlight_green',
}

window_name = 'Yolo_test'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

speaker = Speaker()
last_announced = None
last_time = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print('프레임 읽기 실패')
        break

    result = model.predict(frame, imgsz=416, conf=0.4, verbose=False, device='cpu')
    r = result[0]
    boxes_obj = r.boxes

    announced = None
    if boxes_obj:
        red = False
        green = False
        cross = False

        for b in boxes_obj:
            arr = b.xyxy.cpu().numpy()
            arr = arr.astype(int)
            x1, y1, x2, y2 = arr[0]
            cid = int(b.cls.cpu().numpy()[0])
            cname = model.model.names[cid]
            conf = float(b.conf.cpu().numpy()[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cname}:{conf:.2f}", (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if cid == 0:
                cross = True
            elif cid == 1:
                red = True
            elif cid == 2:
                green = True

        if cross:
            if red:
                announced = '빨간 불입니다. 멈추세요'
            elif green:
                announced = '초록 불입니다. 건너세요'
            else:
                announced = '신호등이 없는 횡단보도입니다. 조심해서 건너세요'


    if announced and (announced != last_announced or time.time() - last_time > 5):
        speaker.say(announced)
        last_announced = announced
        last_time = time.time()

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)

    if key in [ord('q'), 27]:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <1:
        break

cap.release()
cv2.destroyAllWindows()
