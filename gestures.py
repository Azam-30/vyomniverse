import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
 
st.set_page_config(layout="wide")
 
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
 
with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

genai.configure(api_key="AIzaSyDGnPS49ssU5Z0map8ixTwgM6Nj2eQo-BE")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam with a default camera index of 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    if img is not None:
        hands, img = detector.findHands(img, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)
            return fingers, lmList, img
    return None, None, img

def draw(info, prev_pos, canvas):
    if info:
        fingers, lmList = info
        current_pos = None
        if fingers == [0, 1, 0, 0, 0]:
            current_pos = lmList[8][0:2]
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
        elif fingers == [1, 0, 0, 0, 0]:
            canvas = np.zeros_like(canvas)
        return current_pos, canvas
    return prev_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this question", pil_image])
        return response.text if response else ""
    return ""

prev_pos = None
canvas = None
output_text = ""
image_combined = None

while run:
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Retrying...")
        cap.release()
        cap = cv2.VideoCapture(0)  # Try reinitializing the camera
        continue
    
    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info, lmList, img = getHandInfo(img)
    if info:
        fingers, lmList = info, lmList
        prev_pos, canvas = draw((fingers, lmList), prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Exit loop if 'Run' checkbox is unchecked
    if not run:
        break

# Release the camera resource after exiting the loop
cap.release()
cv2.destroyAllWindows()
