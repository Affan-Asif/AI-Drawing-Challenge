import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() 
mpDraw = mp.solutions.drawing_utils

while True:
    success,img = cap.read()

    results = hands.process(img)
    print(results)
    print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            print(i.landmark)
            for id, lm in enumerate(i.landmark):
                h,w,c=img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                if id==4:
                    cv2.circle(img,(cx,cy),15,(255,0,0),-1)

            mpDraw.draw_landmarks(img,i,mpHands.HAND_CONNECTIONS)

    cv2.imshow('img',img)
    cv2.waitKey(1)