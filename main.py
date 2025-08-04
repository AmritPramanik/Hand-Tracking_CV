import cv2
import cvzone
from ultralytics import YOLO

model = YOLO('./Yolo-Weights/yolov8m.pt').to('cuda')
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    result = model(frame)
    annotated_frame = result[0].plot()
    cv2.imshow('YOLO',annotated_frame)

    if cv2.waitKey(1) == ord('x'):
        break
cv2.destroyAllWindows()