import os
import cv2
import numpy as np
import mediapipe as mp

# Paths for accessories and video
IMG_KCMT = os.path.join(os.getcwd(), 'attachment', 'sdg.png')
IMG_CIGAR = os.path.join(os.getcwd(), 'attachment', 'cigar.png')
VID_PATH = os.path.join(os.getcwd(), 'attachment', 'sample-renamed.mp4')

# Video capture and output setup
cap = cv2.VideoCapture(VID_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (frame_width, frame_height))

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load accessories
kcmt = cv2.imread(IMG_KCMT, cv2.IMREAD_UNCHANGED)
cigar = cv2.imread(IMG_CIGAR, cv2.IMREAD_UNCHANGED)

# Validate alpha channels
if kcmt.shape[2] != 4 or cigar.shape[2] != 4:
    raise ValueError("Both accessory images must have alpha channels.")

# Optical flow tracking variables
prev_pts = None
prev_frame_gray = None
tracking_interval = 10  # Reinitialize tracking every 10 frames
frame_count = 0

# Helper functions
def calculate_angle(landmark1, landmark2):
    delta_y = landmark2[1] - landmark1[1]
    delta_x = landmark2[0] - landmark1[0]
    return -np.arctan2(delta_y, delta_x) * 180.0 / np.pi

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    x, y, w, h = map(int, [x, y, w, h])

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

    background_region = background[y:y + h, x:x + w]
    composite = background_region * (1 - alpha) + overlay_colors * alpha
    background[y:y + h, x:x + w] = np.clip(composite, 0, 255).astype(np.uint8)
    return background

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # Reinitialize tracking if needed
    if prev_pts is None or frame_count % tracking_interval == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for flm in results.multi_face_landmarks:
                # Collect landmarks and convert to pixel coordinates
                prev_pts = np.array(
                    [[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in flm.landmark],
                    dtype=np.float32
                )
                prev_frame_gray = frame_gray

    elif prev_pts is not None:
        # Optical flow tracking
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, frame_gray, prev_pts.reshape(-1, 1, 2), None, **lk_params
        )

        # If tracking fails for too many points, reinitialize
        if np.sum(status) < len(status) / 2:
            prev_pts = None
            
        else:
            # Update the tracked points
            prev_pts = next_pts.reshape(-1, 2)
            prev_frame_gray = frame_gray

            # Use tracked points to calculate glasses and cigar positions
            left_outer = tuple(prev_pts[33])
            left_inner = tuple(prev_pts[133])
            right_outer = tuple(prev_pts[362])
            right_inner = tuple(prev_pts[263])
            mouth_left = tuple(prev_pts[61])
            mouth_right = tuple(prev_pts[291])
            mouth_top = tuple(prev_pts[13])
            mouth_bottom = tuple(prev_pts[14])

            # Calculate rotation angles and accessory dimensions
            eye_angle = calculate_angle(left_outer, left_inner)
            cigar_angle = calculate_angle(mouth_left, mouth_right)

            eye_distance = abs(right_outer[0] - left_outer[0])
            glasses_width = int(eye_distance * 1.8)
            aspect_ratio = kcmt.shape[0] / kcmt.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)

            mouth_distance = abs(mouth_right[0] - mouth_left[0])
            cigar_width = int(mouth_distance * 1.8)
            aspect_ratio_cigar = cigar.shape[0] / cigar.shape[1]
            cigar_height = int(cigar_width * aspect_ratio_cigar)

            # Resize and rotate accessories
            kcmt_resized = cv2.resize(kcmt, (glasses_width, glasses_height))
            kcmt_rotated = rotate_image(kcmt_resized, eye_angle)

            cigar_resized = cv2.resize(cigar, (cigar_width, cigar_height))
            cigar_rotated = rotate_image(cigar_resized, cigar_angle)

            # Calculate positions
            center_x = (left_outer[0] + right_outer[0]) // 2
            center_y = (left_outer[1] + right_outer[1]) // 2
            x = center_x - kcmt_rotated.shape[1] // 2
            y = center_y - kcmt_rotated.shape[0] // 2

            center_x_cigar = (mouth_left[0] + mouth_right[0]) // 2
            center_y_cigar = (mouth_bottom[1] + mouth_top[1]) // 2
            x_cigar = center_x_cigar - cigar_rotated.shape[1] // 2 - int(cigar_width * 0.45)
            y_cigar = center_y_cigar - cigar_rotated.shape[0] // 2 + int(cigar_height * 0.45)

            # Overlay the images
            frame = overlay_image(frame, kcmt_rotated, x, y)
            frame = overlay_image(frame, cigar_rotated, x_cigar, y_cigar)

    cv2.imshow('Facial Landmark Detection with Optical Flow', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
