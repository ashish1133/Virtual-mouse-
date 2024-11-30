import cv2
import mediapipe as mp
import pyautogui
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Start Webcam
cap = cv2.VideoCapture(0)
cap_width = 640  # Webcam resolution width
cap_height = 480  # Webcam resolution height

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture video frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Draw landmarks and control mouse
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Map to screen coordinates
            screen_x = int(index_finger_tip.x * screen_width)
            screen_y = int(index_finger_tip.y * screen_height)

            # Move the mouse pointer
            pyautogui.moveTo(screen_x, screen_y)

            # Detect left-click gesture (distance between index and thumb tips)
            left_click_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 +
                                   (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
            if left_click_distance < 0.02:  # Adjust threshold
                pyautogui.click()

            # Detect right-click gesture (distance between thumb and middle finger tips)
            right_click_distance = ((thumb_tip.x - middle_finger_tip.x) ** 2 +
                                     (thumb_tip.y - middle_finger_tip.y) ** 2) ** 0.5
            if right_click_distance < 0.02:  # Adjust threshold
                pyautogui.rightClick()  # Right-click action

    # Display the video frame
    cv2.imshow("Virtual Mouse", frame)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
