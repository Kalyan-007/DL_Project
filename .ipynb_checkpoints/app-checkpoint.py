import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from time import sleep

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# load model
file_name = "D:/DLProject/RF_model_full.pkl"
model = pickle.load(open(file_name,'rb'))

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 4
fontColor = (255,0,0)
thickness = 4
lineType = 2

st.title("DL Project")
run = st.checkbox('run')
FRAME_WINDOW = st.image([])

def print_points(lms,size):
    w, h, d = size
    for i in [8,12,16]:
        x  = lms.landmark[i].x
        y = lms.landmark[i].y
        z = lms.landmark[i].z
        st.write(f"x {x*size[0]} y {y*size[1]} z {z*size[0]} ")
        # print(lms.landmark[i], type(lms.landmark[i]),lms.landmark.x)
    # for id, lm in enumerate(lms.landmark):
    #     px, py = int(lm.x * w), int(lm.y * h)
    # print(lm.landmark[8])
    # print(px)
    return 

def clean(dic):
    len_dic = len(dic.keys())
    features = []
    temp = [0]*42
    for side, item in dic.items():
        for lm in item.landmark:
            features.append(lm.x)
            features.append(lm.y)
    if len(features) == 42 and list(dic.keys())[0] == 'Right':
        features = features + temp 
    if len(features) == 42 and list(dic.keys())[0] == 'Left':
        features = temp + features
#     print(features, len(features))
    return features

# Proper resource cleanup
cap = cv2.VideoCapture(0)

try:
    while run:
        if not cap.isOpened():
            st.error("Could not access the webcam.")
            break

        success, image = cap.read()
        if not success:
            st.warning("No frames available from the camera.")
            continue

        with mp_hands.Hands(model_complexity=0,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            size = image.shape
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                land_marks = {}
                for hand_type, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                    land_marks[hand_type.classification[0].label] = hand_landmarks

                x = np.array(clean(land_marks))
                if x.size == 84:  # Validate feature vector size
                    x_ = x.reshape(1, -1)
                    y = model.predict(x_)
                    label = classes[y[0]]
                    cv2.putText(image, label, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(image)
finally:
    cap.release()
    st.write("Stopped")
