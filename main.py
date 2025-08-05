import cv2
import  mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    ret,frame = cap.read()

    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks :
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==4 :
                    cv2.circle(frame, (cx, cy),15,(255,0,255),thickness=-1)

                print(id, cx, cy)
                if id == 8:
                    cv2.circle(frame, (cx, cy), 15, (255,0,0), thickness=-1)

            custom_connection_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=4)
            mpDraw.draw_landmarks(frame,handlms,mpHands.HAND_CONNECTIONS,connection_drawing_spec=custom_connection_style)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame,str(f"FPS : {int(fps)}"),(10,30),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)

    cv2.imshow('Hand Tracking',frame)
    if cv2.waitKey(1) == ord('x'):
        break
cv2.destroyAllWindows()