import cv2
import mediapipe as mp
import pandas as pd

VIDEO_ID = 1

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not working")
    exit()

columns = [
    "id",
    "left_shoulder_x","left_shoulder_y",
    "left_elbow_x","left_elbow_y",
    "left_wrist_x","left_wrist_y",
    "right_shoulder_x","right_shoulder_y",
    "right_elbow_x","right_elbow_y",
    "right_wrist_x","right_wrist_y",

    "h1_thumb_x","h1_thumb_y",
    "h1_index_x","h1_index_y",
    "h1_middle_x","h1_middle_y",
    "h1_ring_x","h1_ring_y",
    "h1_pinky_x","h1_pinky_y",

    "h2_thumb_x","h2_thumb_y",
    "h2_index_x","h2_index_y",
    "h2_middle_x","h2_middle_y",
    "h2_ring_x","h2_ring_y",
    "h2_pinky_x","h2_pinky_y"
]

df = pd.DataFrame(columns=columns)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    h, w, _ = frame.shape

    left_shoulder = left_elbow = left_wrist = (-1, -1)
    right_shoulder = right_elbow = right_wrist = (-1, -1)

    h1 = [(-1, -1)] * 5
    h2 = [(-1, -1)] * 5

    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        left_shoulder, left_elbow, left_wrist = pt(11), pt(13), pt(15)
        right_shoulder, right_elbow, right_wrist = pt(12), pt(14), pt(16)

        cv2.line(frame, left_shoulder, left_elbow, (0, 255, 0), 3)
        cv2.line(frame, left_elbow, left_wrist, (0, 255, 0), 3)
        cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 3)
        cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 3)

    if hand_results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingertip_ids = [4, 8, 12, 16, 20]
            fingertips = []

            for tip in fingertip_ids:
                x = int(hand_landmarks.landmark[tip].x * w)
                y = int(hand_landmarks.landmark[tip].y * h)
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
                fingertips.append((x, y))

            if hand_index == 0:
                h1 = fingertips
            elif hand_index == 1:
                h2 = fingertips

    row = {
        "id": VIDEO_ID,

        "left_shoulder_x": left_shoulder[0],
        "left_shoulder_y": left_shoulder[1],
        "left_elbow_x": left_elbow[0],
        "left_elbow_y": left_elbow[1],
        "left_wrist_x": left_wrist[0],
        "left_wrist_y": left_wrist[1],

        "right_shoulder_x": right_shoulder[0],
        "right_shoulder_y": right_shoulder[1],
        "right_elbow_x": right_elbow[0],
        "right_elbow_y": right_elbow[1],
        "right_wrist_x": right_wrist[0],
        "right_wrist_y": right_wrist[1],
    }

    names = ["thumb", "index", "middle", "ring", "pinky"]

    for i, name in enumerate(names):
        row[f"h1_{name}_x"] = h1[i][0]
        row[f"h1_{name}_y"] = h1[i][1]

        row[f"h2_{name}_x"] = h2[i][0]
        row[f"h2_{name}_y"] = h2[i][1]

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

df.to_csv("data.csv", index=False)
print("Saved to data.csv")