import os 

from glob import glob
import numpy as np
import mediapipe as mp
import cv2 
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

## Vid Path
VID_PATH = os.path.join(os.getcwd(), "sample.mp4")

## Inisialisasi face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

## Inisialissi Drwaing utility
mp_drawing = mp.solutions.drawing_utils

## menentukan landmark yang akan diambil
left_eye_x1 = 70
left_eye_x2 = 188
right_eye_x1 = 285
right_eye_x2 = 261

## Open video
cap = cv2.VideoCapture(VID_PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ## Konversoiung ke RGB
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # results = face_mesh.process(rgb_frame)

    ## Extrak landmark mata

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = face_mesh.process(frame)
    # if results.multi_face_landmarks:
    #     for face_landmarks in results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)
    #         landmarks = face_landmarks.landmark
    #         left_eye = landmarks[left_eye_x1:left_eye_x2]
    #         right_eye = landmarks[right_eye_x1:right_eye_x2]
    #         left_eye = np.array([(lm.x, lm.y) for lm in left_eye])
    #         right_eye = np.array([(lm.x, lm.y) for lm in right_eye])
    #         break
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cap.destroyAllWindows()