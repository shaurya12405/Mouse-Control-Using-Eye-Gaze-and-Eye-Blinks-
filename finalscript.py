# Importing Necessary Libraries 
import cv2
import numpy as np
import mediapipe as mp     # Google's Model to detect Face and Hand Gestures
import pyautogui           # Helps in automating the Mouse and Keyboard
import time
from PIL import ImageGrab  # Helps Take the Screenshot
from math import dist       

# Configuration constants
CAM_WIDTH = 640
CAM_HEIGHT = 360
FRAME_SKIP = 1
SMOOTHING_FACTOR = 0.5      # Determines how smooth the cursor moves
EAR_THRESHOLD = 0.22        # It determines the sensitivity of the Blinks (Lower value implies Stricter)
USE_GPU = False

# Initializing Mediapipe Models
mp_face_mesh = mp.solutions.face_mesh       # Detects 468 Facial Landmarks
mp_hands = mp.solutions.hands               # Detects 21 Hand Landmarks per Hand 
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6, static_image_mode=False)

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Eye and facial landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
SCROLL_THRESH = 0.55
SMILE_THRESH = 3

# State variables
frame_counter = 0
smoothed_x, smoothed_y = screen_w // 2, screen_h // 2
last_actions = []

# Blink detection state
ear_history = {'left': [], 'right': []}
blink_counter = {'left': 0, 'right': 0}
MIN_CONSEC_FRAMES = 2
SLOPE_THRESHOLD = 0.01
last_blink_times = {'left': 0, 'right': 0, 'both': 0}

def calculate_EAR(eye_points, landmarks, image_shape):
    h, w = image_shape
    horizontal = abs(landmarks[eye_points[3]].x - landmarks[eye_points[0]].x) * w
    vertical1 = dist((landmarks[eye_points[1]].x*w, landmarks[eye_points[1]].y*h),
                     (landmarks[eye_points[5]].x*w, landmarks[eye_points[5]].y*h))
    vertical2 = dist((landmarks[eye_points[2]].x*w, landmarks[eye_points[2]].y*h),
                     (landmarks[eye_points[4]].x*w, landmarks[eye_points[4]].y*h))
    return (vertical1 + vertical2) / (2.0 * horizontal)

def detect_blink(left_ear, right_ear, last_times, ear_history, blink_counter):
    current_time = time.time()
    actions = []

    ear_history['left'].append(left_ear)
    ear_history['right'].append(right_ear)
    if len(ear_history['left']) > 3:
        ear_history['left'].pop(0)
    if len(ear_history['right']) > 3:
        ear_history['right'].pop(0)

    left_slope = ear_history['left'][-2] - left_ear if len(ear_history['left']) >= 2 else 0
    right_slope = ear_history['right'][-2] - right_ear if len(ear_history['right']) >= 2 else 0

    left_closed = left_ear < EAR_THRESHOLD or left_slope > SLOPE_THRESHOLD
    right_closed = right_ear < EAR_THRESHOLD or right_slope > SLOPE_THRESHOLD

    if left_closed and right_closed:
        blink_counter['left'] += 1
        blink_counter['right'] += 1
        if blink_counter['left'] >= MIN_CONSEC_FRAMES and blink_counter['right'] >= MIN_CONSEC_FRAMES:
            if current_time - last_times['both'] > 4:
                pyautogui.doubleClick()
                actions.append("Double Click")
                last_times.update({'both': current_time, 'left': current_time, 'right': current_time})
                blink_counter['left'] = 0
                blink_counter['right'] = 0
    else:
        if left_closed:
            blink_counter['left'] += 1
        elif blink_counter['left'] >= MIN_CONSEC_FRAMES and current_time - last_times['left'] > 0.05:
            pyautogui.click()
            actions.append("Left Click")
            last_times['left'] = current_time
            blink_counter['left'] = 0
        else:
            blink_counter['left'] = 0

        if right_closed:
            blink_counter['right'] += 1
        elif blink_counter['right'] >= MIN_CONSEC_FRAMES and current_time - last_times['right'] > 0.05:
            pyautogui.rightClick()
            actions.append("Right Click")
            last_times['right'] = current_time
            blink_counter['right'] = 0
        else:
            blink_counter['right'] = 0

    return actions, last_times, ear_history, blink_counter

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

print("Optimized gesture control running. Press 'q' to exit.")

while True:
    frame_counter += 1
    if frame_counter % FRAME_SKIP != 0:
        continue

    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    current_actions = []

    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark

        nose = landmarks[4]
        smoothed_x = SMOOTHING_FACTOR * smoothed_x + (1 - SMOOTHING_FACTOR) * nose.x
        smoothed_y = SMOOTHING_FACTOR * smoothed_y + (1 - SMOOTHING_FACTOR) * nose.y
        pyautogui.moveTo(int(smoothed_x * screen_w), int(smoothed_y * screen_h))

        left_ear = calculate_EAR(LEFT_EYE, landmarks, (h, w))
        right_ear = calculate_EAR(RIGHT_EYE, landmarks, (h, w))
        actions, last_blink_times, ear_history, blink_counter = detect_blink(
            left_ear, right_ear, last_blink_times, ear_history, blink_counter)
        current_actions.extend(actions)

        head_tilt = landmarks[10].y - landmarks[152].y
        if abs(head_tilt) > SCROLL_THRESH:
            pyautogui.scroll(int(head_tilt * 100))
            current_actions.append("Scroll")

        mouth_width = abs(landmarks[61].x - landmarks[291].x)
        if mouth_width > SMILE_THRESH:
            pyautogui.hotkey('ctrl', 'tab')
            current_actions.append("Tab Switch")

    if hand_results.multi_hand_landmarks:
        hand = hand_results.multi_hand_landmarks[0]
        fingers = sum([
            hand.landmark[4].x < hand.landmark[3].x,
            hand.landmark[8].y < hand.landmark[6].y,
            hand.landmark[12].y < hand.landmark[10].y,
            hand.landmark[16].y < hand.landmark[14].y,
            hand.landmark[20].y < hand.landmark[18].y
        ])

        if fingers == 0:
            pyautogui.hotkey('win', 'ctrl', 'o')
            current_actions.append("Keyboard")
        elif fingers == 1:
            pyautogui.hotkey('ctrl', 'v')
            current_actions.append("Paste")
        elif fingers == 2:
            pyautogui.hotkey('ctrl', 'c')
            current_actions.append("Copy")
        elif fingers == 3:
            pyautogui.hotkey('ctrl', 'z')
            current_actions.append("Undo")
        elif fingers == 5:
            ImageGrab.grab().save("screenshot.png")
            current_actions.append("Screenshot")

    last_actions = (last_actions + current_actions)[-3:]
    if last_actions:
        cv2.putText(frame, " | ".join(last_actions), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
