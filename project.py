import cv2
import mediapipe as mp
import numpy as np

from PIL import Image
import google.generativeai as genai

# Gemini API setup
genai.configure(api_key="AIzaSyBIU-wH9KCx04jiCXTe69HfBdYou4KKehA")
model = genai.GenerativeModel('gemini-1.5-flash')

cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpHands = mp.solutions.hands
hands= mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

canvas =None
prev_pos=None

def fingersup(lm):
    fingers = []
    fingers.append(0) #thumb upp
    fingers.append(1 if lm[8][1]<lm[6][1] - 10 else 0) #thumb upp
    fingers.append(1 if lm[12][1]<lm[10][1] - 10 else 0) #thumb upp
    fingers.append(1 if lm[16][1]<lm[14][1] - 10 else 0) #thumb upp
    fingers.append(1 if lm[20][1]<lm[18][1] - 10 else 0) #thumb upp

    return fingers


while True:
    success,img =cap.read()
    img = cv2.flip(img,1)

    if canvas is None:
        canvas=np.zeros_like(img)

    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results= hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        lm_list=[]
        h,w,c= img.shape
        for lm in hand.landmark:
            cx,cy = int(lm.x*w),int(lm.y*h)
            lm_list.append((cx,cy))

        fingers_state = fingersup(lm_list)#[0,0,0,0,0]
        print(fingers_state)

        if fingers_state==[0,1,0,0,0]:
            curr_pos = lm_list[8]
            if prev_pos is None:
                prev_pos = curr_pos
            cv2.line(canvas,prev_pos,curr_pos,(255,0,255),10)
            prev_pos=curr_pos
        else:
            prev_pos=None

        if fingers_state==[0,0,0,0,0]:
            canvas = np.zeros_like(img)

        
        if fingers_state==[0,1,1,1,0]:
            pil_image = Image.fromarray(canvas)
            response = model.generate_content([
                "Solve this math problem:", 
                pil_image
            ])
            print("Gemini Response:", response.text)


    combined = cv2.addWeighted(img,0.7,canvas,0.3,0)
    cv2.imshow("Draw",combined)
    cv2.waitKey(1)            



