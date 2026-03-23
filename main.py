import cv2
import mediapipe as mp
import pandas as pd
import time

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
frame_count = 0
start_time = time.time()
duration = 3  # seconds

# Prepare an empty DataFrame
columns = [
    "frame",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "hand1_fingertips", "hand2_fingertips"
]
df = pd.DataFrame(columns=columns)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - start_time >= duration:
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    h, w, _ = frame.shape

    # Default values
    left_shoulder = left_elbow = left_wrist = (-1, -1)
    right_shoulder = right_elbow = right_wrist = (-1, -1)
    hand1_fingertips = []
    hand2_fingertips = []

    # Arms
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        def pt(i): return int(lm[i].x * w), int(lm[i].y * h)

        left_shoulder, left_elbow, left_wrist = pt(11), pt(13), pt(15)
        right_shoulder, right_elbow, right_wrist = pt(12), pt(14), pt(16)

        # Draw arms
        cv2.line(frame, left_shoulder, left_elbow, (0, 255, 0), 3)
        cv2.line(frame, left_elbow, left_wrist, (0, 255, 0), 3)
        cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 3)
        cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 3)
        for point in [left_shoulder, left_elbow, left_wrist,
                      right_shoulder, right_elbow, right_wrist]:
            cv2.circle(frame, point, 8, (0, 0, 255), -1)

    # Hands
    if hand_results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingertip_ids = [4, 8, 12, 16, 20]
            fingertips = []
            for tip in fingertip_ids:
                x = int(hand_landmarks.landmark[tip].x * w)
                y = int(hand_landmarks.landmark[tip].y * h)
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                fingertips.append((x, y))

            if hand_index == 0:
                hand1_fingertips = fingertips
            elif hand_index == 1:
                hand2_fingertips = fingertips

    # Append row to DataFrame
    df = pd.concat([df, pd.DataFrame([{
        "frame": frame_count,
        "left_shoulder": left_shoulder,
        "left_elbow": left_elbow,
        "left_wrist": left_wrist,
        "right_shoulder": right_shoulder,
        "right_elbow": right_elbow,
        "right_wrist": right_wrist,
        "hand1_fingertips": hand1_fingertips,
        "hand2_fingertips": hand2_fingertips
    }])], ignore_index=True)

    cv2.imshow("Arms + Hands Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save DataFrame to CSV
df.to_csv("videos\\data.csv", index=False)
print("Saved positions to arm_hand_positions.csv")