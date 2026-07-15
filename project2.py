import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import streamlit as st
import google.generativeai as genai

# Set Gemini API key
genai.configure(api_key="AIzaSyBIU-wH9KCx04jiCXTe69HfBdYou4KKehA")
model = genai.GenerativeModel('gemini-1.5-flash')

# Streamlit UI setup
st.set_page_config(page_title="Gesture-Based Math Solver", layout="wide")
st.title("✋ Gesture-Based Math Solver with Gemini")
frame_placeholder = st.empty()
response_placeholder = st.empty()

# Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_pos = None

def fingersup(lm):
    fingers = []
    fingers.append(0)  # Thumb not checked here
    fingers.append(1 if lm[8][1] < lm[6][1] - 10 else 0)   # Index
    fingers.append(1 if lm[12][1] < lm[10][1] - 10 else 0) # Middle
    fingers.append(1 if lm[16][1] < lm[14][1] - 10 else 0) # Ring
    fingers.append(1 if lm[20][1] < lm[18][1] - 10 else 0) # Pinky
    return fingers

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera")
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm_list = []
        h, w, c = frame.shape
        for lm in hand.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))

        fingers_state = fingersup(lm_list)

        if fingers_state == [0, 1, 0, 0, 0]:  # Drawing mode
            curr_pos = lm_list[8]
            if prev_pos is None:
                prev_pos = curr_pos
            cv2.line(canvas, prev_pos, curr_pos, (255, 0, 255), 10)
            prev_pos = curr_pos
        else:
            prev_pos = None

        if fingers_state == [0, 0, 0, 0, 0]:  # Clear canvas
            canvas = np.zeros_like(frame)

        if fingers_state == [0, 1, 1, 1, 0]:  # Send to Gemini
            pil_image = Image.fromarray(canvas)
            response_placeholder.markdown("🧠 Analyzing with Gemini...")
            try:
                response = model.generate_content([
                    "Solve this math problem:", 
                    pil_image
                ])
                response_placeholder.markdown(f"### Gemini Response:\n{response.text}")
            except Exception as e:
                response_placeholder.error(f"Gemini API error: {str(e)}")

    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(combined, channels="RGB")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
