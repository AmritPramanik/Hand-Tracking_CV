import cv2
import  mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    ret,frame = cap.read()

    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks :
        for handlms in results.multi_hand_landmarks:
            custom_connection_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
            mpDraw.draw_landmarks(frame,handlms,mpHands.HAND_CONNECTIONS,connection_drawing_spec=custom_connection_style)


    cv2.imshow('Hand Tracking',frame)
    if cv2.waitKey(1) == ord('x'):
        break
cv2.destroyAllWindows()