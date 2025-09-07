import queue
import threading

import cv2
import platform
from ultralytics import YOLO
from collections import deque

import pyttsx3
import pythoncom
import time

import re
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# tts
class Speaker:
    def __init__(self, lang='ko', rate='175'):
        self.q = queue.Queue(maxsize=1)
        self.lang = lang
        self.rate = int(rate)
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
        while True:
            text = self.q.get()
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                for v in engine.getProperty("voices"):
                    vid = (v.id or "").lower()
                    vname = (v.name or "").lower()
                    if "ko" in vid or "korean" in vname:
                        engine.setProperty("voice", v.id)
                        break
                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception as e:
                print("[TTS ERROR]", e)

# smoothing 작업
class StateSmoother:
    def __init__(self, window=10, min_frames=4, ratio=0.6):
        self.history = deque(maxlen=window)
        self.min_frames = min_frames
        self.ratio = ratio

    def push(self, cross, red, green):
        self.history.append((cross, red, green))

    def get_stable(self):
        n = len(self.history)
        if n == 0:
            return False, None
        c = sum(1 for h in self.history if h[0])
        r = sum(1 for h in self.history if h[1])
        g = sum(1 for h in self.history if h[2])
        cross_stable = (c >= self.min_frames) and (c / n >= self.ratio)
        light = None
        if (r >= self.min_frames) and (r / n >= self.ratio):
            light = 'red'
        elif (g >= self.min_frames) and (g / n >= self.ratio):
            light = 'green'
        return cross_stable, light

class BoxSmoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.prev = {}

    def update(self, key, box):
        if box is None:
            self.prev.pop(key, None)
            return None
        p = self.prev.get(key)
        if p is None:
            cur = tuple(int(v) for v in box)
        else:
            cur = tuple(int(self.alpha * box[i] + (1 - self.alpha) * p[i]) for i in range(4))
        self.prev[key] = cur
        return cur

def crop_box(frame, box, pad=[10, 100, 10, 100]):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    nx1 = max(0, x1 - pad[0])
    ny1 = max(0, y1 - pad[1])
    nx2 = min(w, x2 + pad[2])
    ny2 = min(h, y2 + pad[3])
    crop = frame[ny1:ny2, nx1:nx2]

    return crop, (nx1,ny1,nx2,ny2)

def digit_read(frame, color):

    if color == 'red':
        g = frame[:, :, 2]
    elif color == 'green':
        g = frame[:, :, 1]

    g = cv2.equalizeHist(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    _, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    k = np.ones((2, 2), np.uint8)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, k, 1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, k, 1)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, k, 1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, k, 1)

    th1 = cv2.resize(th1, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    th2 = cv2.resize(th2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

    cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"

    for th in (th1, th2):
        txt = pytesseract.image_to_string(th, config=cfg)
        m = re.search(r"(\d+)", txt)
        if not m:
            continue
        v = int(m.group(1))
        if 0 <= v <= 99:
            return v
    return None


# os 구분(노트북 : 시험용, 라즈베리파이 : 실전용)
os = platform.system().lower()
if os == 'windows':
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
elif os == 'linux':
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
else:
    raise TypeError('os를 인식할 수 없습니다')

# model 위치
model = YOLO("D:/dev/navis/models/best2.pt")
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
last_light = None
last_seen = 0
no_signal_wait = 1.2
cross_since = 0.0

smoother = StateSmoother(window=10, min_frames=4, ratio=0.6)
box_smoother = BoxSmoother(alpha=0.4)

# 메인 코드
while True:
    ok, frame = cap.read()
    if not ok:
        print('프레임 읽기 실패')
        break

    result = model.predict(frame, imgsz=416, conf=0.2, verbose=False, device='cpu')
    r = result[0]
    boxes_obj = r.boxes

    cross = False
    red = False
    green = False

    best = {'crosswalk': None, 'trafficlight_red': None, 'trafficlight_green': None}
    best_conf = {'crosswalk': -1.0, 'trafficlight_red': -1.0, 'trafficlight_green': -1.0}
    if boxes_obj:

        for b in boxes_obj:
            arr = b.xyxy.cpu().numpy()
            arr = arr.astype(int)
            x1, y1, x2, y2 = arr[0]
            cid = int(b.cls.cpu().numpy()[0])
            cname = model.model.names[cid]
            conf = float(b.conf.cpu().numpy()[0])

            if cid == 0:
                cross = True
                if conf > best_conf['crosswalk']:
                    best_conf['crosswalk'] = conf
                    best['crosswalk'] = (x1, y1, x2, y2)

            elif cid == 1:
                red = True
                if conf > best_conf['trafficlight_red']:
                    best_conf['trafficlight_red'] = conf
                    best['trafficlight_red'] = (x1, y1, x2, y2)

            elif cid == 2:
                green = True
                if conf > best_conf['trafficlight_green']:
                    best_conf['trafficlight_green'] = conf
                    best['trafficlight_green'] = (x1, y1, x2, y2)

    smoother.push(cross, red, green)
    cross_stable, light = smoother.get_stable()

    now = time.time()

    if cross_stable:
        if cross_since == 0.0:
            cross_since = now
    else:
        cross_since = 0.0

    if light is not None:
        last_light = light
        last_seen = now
    else:
        if now - last_seen < 1.2:
            light = last_light

    suppress_no_signal = False
    if cross_stable and (light is None):
        if cross_since > 0.0 and (now - cross_since) < no_signal_wait:
            suppress_no_signal = True

    num = None
    if cross_stable and light in ("red", "green"):
        key = "trafficlight_red" if light == "red" else "trafficlight_green"
        box = best.get(key)
        if box:
            crop, _ = crop_box(frame, box)
            num = digit_read(crop, light)

    announced = None
    if cross_stable:
        if light == "red":
            announced = (f"빨간 불입니다. {num}초 남았습니다. 멈추세요"
                         if num is not None else "빨간 불입니다. 멈추세요")
        elif light == "green":
            announced = (f"초록 불입니다. {num}초 남았습니다. 조심하세요"
                         if num is not None else "초록 불입니다. 조심하세요")
        else:
            if not suppress_no_signal:
                announced = "신호등이 없는 횡단보도입니다. 조심하세요"

    if announced and (announced != last_announced or time.time() - last_time > 5):
        speaker.say(announced)
        last_announced = announced
        last_time = time.time()

    color_map = {
        'crosswalk': (255, 128, 0),
        'trafficlight_red': (0, 0, 255),
        'trafficlight_green': (0, 200, 0),
    }

    for k in ('crosswalk', 'trafficlight_red', 'trafficlight_green'):
        smoothed = box_smoother.update(k, best[k])
        if smoothed:
            x1, y1, x2, y2 = smoothed
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[k], 2)
            label = f"{k}"
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map[k], 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)

    if key in [ord('q'), 27]:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <1:
        break

cap.release()
cv2.destroyAllWindows()
