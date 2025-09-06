import sys
import cv2
from ultralytics import YOLO

model = YOLO(r"D:\deocent\dev\best.pt")  # 경로는 raw string 추천

# 카메라 1번을 DirectShow 백엔드로 오픈
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 웹캠 오류 처리
if not cap.isOpened():
    print("camera open failed (index=1, backend=CAP_DSHOW)")
    sys.exit(1)

while True:
    ok, frame = cap.read()
    if not ok:
        print("frame read failed")
        break

    # YOLO 추론 및 시각화
    results = model.predict(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("YOLO (cam 1, DSHOW)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
