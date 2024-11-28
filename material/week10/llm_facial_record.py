# Import
import os
import cv2
import numpy as np
import mediapipe as mp

# Lokasi gambar kacamata
IMG_KCMT = os.path.join(os.getcwd(), 'attachment', 'sdg.png')
IMG_CIGAR = os.path.join(os.getcwd(), 'attachment', 'cigar.png')

# # Initialize video capture
VID_PATH = os.path.join(os.getcwd(), 'attachment', 'sample-renamed.mp4')

cap = cv2.VideoCapture(VID_PATH)

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (frame_width, frame_height))

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,        # False untuk video, True untuk gambar/foto
    max_num_faces=1,                # Jumlah maksimal wajah yang dideteksi
    min_detection_confidence=0.7,   # Tingkat kepercayaan deteksi wajah
    min_tracking_confidence=0.7     # Tingkat kepercayaan pelacakan wajah
)

# Load gambar kacamata
kcmt = cv2.imread(IMG_KCMT, cv2.IMREAD_UNCHANGED)
cigar = cv2.imread(IMG_CIGAR, cv2.IMREAD_UNCHANGED)

## Tracking variables
frame_count = 0
tracking_interval = 10 # Track every 10 frames

# Check alpha channel
if kcmt.shape[2] != 4:
    raise ValueError('Gambar kacamata tidak memiliki alpha channel')
if cigar.shape[2] != 4:
    raise ValueError('Gambar cerutu tidak memiliki alpha channel')

# Helper Functions
def calculate_angle(landmark1, landmark2):
    """
    Calculate the angle between two landmarks in degrees.
    """
    delta_y = landmark2[1] - landmark1[1]
    delta_x = landmark2[0] - landmark1[0]
    angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi
    return -angle

def rotate_image(image, angle):
    """
    Rotate an image by a given angle around its center.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    return rotated_image

def overlay_image(background, overlay, x, y):
    """
    Overlay an RGBA image onto a background image at position (x, y)
    """
    h, w = overlay.shape[:2]
    
    if x >= background.shape[1] or y >= background.shape[0]:
        return background
    
    if x + w > background.shape[1]:
        w = background.shape[1] - x
    if y + h > background.shape[0]:
        h = background.shape[0] - y
    
    if w <= 0 or h <= 0:
        return background
    
    overlay_colors = overlay[:h, :w, :3]
    alpha = overlay[:h, :w, 3] / 255.0
    alpha = np.dstack((alpha, alpha, alpha))
    
    background_region = background[y:y+h, x:x+w]
    composite = background_region * (1 - alpha) + overlay_colors * alpha
    
    result = background.copy()
    result[y:y+h, x:x+w] = composite
    return result

# Membuka webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    
    tinggi, lebar, _ = frame.shape
    
    if results.multi_face_landmarks:
        for flm in results.multi_face_landmarks:
            ## Eye Landmark
            left_outer = flm.landmark[33]
            left_inner = flm.landmark[133]
            right_outer = flm.landmark[362]
            right_inner = flm.landmark[263]
            
            ## Mouth Landmark
            mouth_left = flm.landmark[61]
            mouth_right = flm.landmark[291]
            mouth_top = flm.landmark[13]
            mouth_bottom = flm.landmark[14]

            # Convert landmarks to pixel coordinates
            left_outer = (int(left_outer.x * lebar), int(left_outer.y * tinggi))
            left_inner = (int(left_inner.x * lebar), int(left_inner.y * tinggi))
            right_outer = (int(right_outer.x * lebar), int(right_outer.y * tinggi))
            right_inner = (int(right_inner.x * lebar), int(right_inner.y * tinggi))
            
            # Convert mouth landmarks to pixel coordinates
            mouth_left = (int(mouth_left.x * lebar), int(mouth_left.y * tinggi))
            mouth_right = (int(mouth_right.x * lebar), int(mouth_right.y * tinggi))
            mouth_top = (int(mouth_top.x * lebar), int(mouth_top.y * tinggi))
            mouth_bottom = (int(mouth_bottom.x * lebar), int(mouth_bottom.y * tinggi))

            # Calculate rotation angle using left eye corner points
            eye_angle = calculate_angle(left_outer, left_inner)

            # Calculate Rotation Angle for Cigar based on the left and right
            cigar_angle = calculate_angle(mouth_left, mouth_right)
            
            # Calculate glasses dimensions
            eye_distance = abs(right_outer[0] - left_outer[0])
            glasses_width = int(eye_distance * 1.8)
            aspect_ratio = kcmt.shape[0] / kcmt.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)

            ## Calculate cigar dimension
            mouth_distance = abs(mouth_right[0] - mouth_left[0])
            cigar_width = int(mouth_distance * 1.8)
            aspect_ratio_cigar = cigar.shape[0] / cigar.shape[1]
            cigar_height = int(cigar_width * aspect_ratio_cigar)

            ## Resize and Rotate the Cigar
            cigar_resized = cv2.resize(cigar, (cigar_width, cigar_height))
            cigar_rotated = rotate_image(cigar_resized, cigar_angle)

            ## Scale the glasses
            glasses_width = int(glasses_width * 1.8)
            glasses_height = int(glasses_height * 1.8)
            
            # Resize and rotate glasses
            kcmt_resized = cv2.resize(kcmt, (glasses_width, glasses_height))
            kcmt_rotated = rotate_image(kcmt_resized, eye_angle)

            # Calculate glasses position
            center_x = (left_outer[0] + right_outer[0]) // 2
            center_y = (left_outer[1] + right_outer[1]) // 2
            x = center_x - kcmt_rotated.shape[1] // 2 + int(glasses_width * 0.05)
            y = center_y - kcmt_rotated.shape[0] // 2
            
            # Calculate cigar position
            center_x_cigar = (mouth_left[0] + mouth_right[0]) // 2
            center_y_cigar = (mouth_bottom[1] + mouth_top[1]) // 2
            x_cigar = center_x_cigar - cigar_rotated.shape[1] // 2 - int(cigar_width * 0.45)
            y_cigar = center_y_cigar - cigar_rotated.shape[0] // 2 + int(cigar_height * 0.45)

            # Overlay rotated cigar on the frame
            frame = overlay_image(frame, cigar_rotated, x_cigar, y_cigar)

            # Overlay rotated glasses on the frame
            frame = overlay_image(frame, kcmt_rotated, x, y)
    
    cv2.imshow('Facial Landmark Detection', frame)

    ## Write the frame to the output video

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
