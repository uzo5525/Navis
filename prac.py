import queue
import threading
import cv2
import platform
from ultralytics import YOLO
import numpy as np
import pyttsx3
import time
import pythoncom
import tempfile, os, winsound

# ------------------ Speaker 클래스 ------------------
class Speaker:
    def __init__(self, rate=175, volume=1.0):
        self.q = queue.Queue(maxsize=1)
        self.rate = rate
        self.volume = volume
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text: str) -> bool:
        if self.q.full():
            try:
                _ = self.q.get_nowait()  # 오래된 거 버림
            except queue.Empty:
                pass
        try:
            self.q.put_nowait(text)
            print(f"[DEBUG] enqueue: {text}")
            return True
        except queue.Full:
            return False

    def _loop(self):
        pythoncom.CoInitialize()
        engine = pyttsx3.init('sapi5')
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)

        # 한국어 보이스 우선 선택
        for v in engine.getProperty('voices'):
            if 'ko' in (v.id+v.name).lower():
                engine.setProperty('voice', v.id)
                break

        while True:
            text = self.q.get()
            try:
                tmp = os.path.join(tempfile.gettempdir(), "tts_tmp.wav")
                engine.save_to_file(text, tmp)
                engine.runAndWait()
                print(f"[DEBUG] file saved: {tmp}")
                winsound.PlaySound(tmp, winsound.SND_FILENAME)
                print(f"[DEBUG] spoken: {text}")
            except Exception as e:
                print("[TTS] error:", e)

# ------------------ 카메라 초기화 ------------------
osname = platform.system().lower()
if osname == 'windows':
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
elif osname == 'linux':
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
else:
    raise TypeError('os를 인식할 수 없습니다')

# ------------------ YOLO 모델 ------------------
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

# ------------------ 메인 루프 ------------------
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
        for b in boxes_obj:
            arr = b.xyxy.cpu().numpy().astype(int)
            x1, y1, x2, y2 = arr[0]
            cid = int(b.cls.cpu().numpy()[0])
            cname = model.model.names[cid]
            conf = float(b.conf.cpu().numpy()[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cname}:{conf:.2f}", (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if cid == 0:
                announced = '횡단보도입니다'
            elif cid == 1:
                announced = '빨간불입니다'
            elif cid == 2:
                announced = '초록불입니다'

    # 3초 쿨다운 로직
    now = time.time()
    if announced and (announced != last_announced or (now - last_time) > 3):
        if speaker.say(announced):  # ✅ say 성공 시에만 갱신
            last_announced = announced
            last_time = now
            print(f"[TTS] {announced} (cooldown {now - last_time:.2f}s)")

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)
    if key in [ord('q'), 27]:
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
