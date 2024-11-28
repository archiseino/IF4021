import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the glasses image (ensure it has a transparent background)
glasses_image = cv2.imread('attachment/sdg.png', -1)  # Load with alpha channel

if glasses_image is None or glasses_image.shape[2] != 4:
    raise ValueError("Glasses image not loaded or missing alpha channel")

# Key point landmarks for eye edges
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

# Function to overlay an image with transparency
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x >= background.shape[1] or y >= background.shape[0]:
        return background

    # Clip the overlay if it goes outside the background
    w = min(w, background.shape[1] - x)
    h = min(h, background.shape[0] - y)
    if w <= 0 or h <= 0:
        return background

    overlay_colors = overlay[:h, :w, :3]
    alpha = overlay[:h, :w, 3] / 255.0
    alpha = np.dstack((alpha, alpha, alpha))

    background_region = background[y:y + h, x:x + w]
    composite = background_region * (1 - alpha) + overlay_colors * alpha
    result = background.copy()
    result[y:y + h, x:x + w] = composite

    return result

# Webcam video feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get key points for both eyes
            left_outer = face_landmarks.landmark[LEFT_EYE_OUTER]
            left_inner = face_landmarks.landmark[LEFT_EYE_INNER]
            right_outer = face_landmarks.landmark[RIGHT_EYE_OUTER]
            right_inner = face_landmarks.landmark[RIGHT_EYE_INNER]

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            left_outer = (int(left_outer.x * w), int(left_outer.y * h))
            left_inner = (int(left_inner.x * w), int(left_inner.y * h))
            right_outer = (int(right_outer.x * w), int(right_outer.y * h))
            right_inner = (int(right_inner.x * w), int(right_inner.y * h))

            # Calculate center between the eyes
            eye_center_x = (left_inner[0] + right_inner[0]) // 2
            eye_center_y = (left_inner[1] + right_inner[1]) // 2

            # Calculate distance between eyes for scaling
            eye_distance = np.linalg.norm(np.array(right_inner) - np.array(left_inner))

            # Resize glasses
            glasses_width = int(eye_distance * 2)  # Adjust for proper fit
            aspect_ratio = glasses_image.shape[0] / glasses_image.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)
            resized_glasses = cv2.resize(glasses_image, (glasses_width, glasses_height))

            # Calculate top-left corner for overlay
            top_left_x = eye_center_x - glasses_width // 2
            top_left_y = eye_center_y - glasses_height // 2 - int(glasses_height * 0.2)

            cv2.rectangle(frame, left_outer, left_inner, (0, 0, 255), 2)
            ## BBOX Mata Kanan
            cv2.rectangle(frame, right_outer, right_inner, (0, 255, 0), 2)

            # Overlay glasses on the frame
            frame = overlay_image(frame, resized_glasses, top_left_x, top_left_y)

    cv2.imshow('Glasses Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
