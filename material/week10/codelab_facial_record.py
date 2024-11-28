import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Define the eye landmark indices (Left and Right Eye landmarks)
LEFT_EYE = list(range(33, 133))  # Left eye (index 33-132)
RIGHT_EYE = list(range(133, 233))  # Right eye (index 133-232)

# Load the glasses image (ensure it has a transparent background)
glasses_image = cv2.imread('attachment/sdg.png', -1)  # Load with alpha channel (transparency)

# Function to extract landmarks for the eyes
def get_eye_landmarks(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = face_mesh.process(image_rgb)
    
    left_eye, right_eye = None, None
    if results.multi_face_landmarks:
        for landmarks_list in results.multi_face_landmarks:
            left_eye = [landmarks_list.landmark[i] for i in LEFT_EYE]
            right_eye = [landmarks_list.landmark[i] for i in RIGHT_EYE]
            
    return left_eye, right_eye

# Function to get the bounding box for the eyes
def get_eye_bounding_box(eye_landmarks):
    min_x = min([landmark.x for landmark in eye_landmarks])
    max_x = max([landmark.x for landmark in eye_landmarks])
    min_y = min([landmark.y for landmark in eye_landmarks])
    max_y = max([landmark.y for landmark in eye_landmarks])
    
    return (min_x, min_y), (max_x, max_y)

# Function to overlay the glasses image
def overlay_glasses(image, eye_box, glasses_image):
    # Convert eye box coordinates to pixel values (assuming image is in pixels)
    h, w, _ = image.shape
    left_top = (int(eye_box[0][0] * w), int(eye_box[0][1] * h))
    right_bottom = (int(eye_box[1][0] * w), int(eye_box[1][1] * h))
    
    # Resize the glasses image to fit the bounding box
    glasses_width = right_bottom[0] - left_top[0]
    glasses_height = right_bottom[1] - left_top[1]
    glasses_resized = cv2.resize(glasses_image, (glasses_width, glasses_height))
    
    # Get the region of interest (ROI) on the face image
    roi = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    
    # Add the glasses image to the face image (handle transparency)
    for c in range(0, 3):  # Loop through the color channels (RGB)
        roi[:, :, c] = roi[:, :, c] * (1 - glasses_resized[:, :, 3] / 255.0) + \
                       glasses_resized[:, :, 3] / 255.0 * glasses_resized[:, :, c]
    
    return image

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the eye landmarks for the current frame
    left_eye_landmarks, right_eye_landmarks = get_eye_landmarks(frame)
    
    # If landmarks are found, overlay glasses
    if left_eye_landmarks:
        left_eye_box = get_eye_bounding_box(left_eye_landmarks)
        frame = overlay_glasses(frame, left_eye_box, glasses_image)
    
    if right_eye_landmarks:
        right_eye_box = get_eye_bounding_box(right_eye_landmarks)
        frame = overlay_glasses(frame, right_eye_box, glasses_image)
    
    # Display the frame with glasses overlayed
    cv2.imshow('Video with Glasses Overlay', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
