import cv2
import mediapipe as mp
import pandas as pd

VIDEO_ID = 1

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

columns = [
    "id",
    "left_shoulder_x","left_shoulder_y",
    "left_elbow_x","left_elbow_y",
    "left_wrist_x","left_wrist_y",
    "right_shoulder_x","right_shoulder_y",
    "right_elbow_x","right_elbow_y",
    "right_wrist_x","right_wrist_y",
    "h1_thumb_x","h1_thumb_y","h1_index_x","h1_index_y",
    "h1_middle_x","h1_middle_y","h1_ring_x","h1_ring_y",
    "h1_pinky_x","h1_pinky_y",
    "h2_thumb_x","h2_thumb_y","h2_index_x","h2_index_y",
    "h2_middle_x","h2_middle_y","h2_ring_x","h2_ring_y",
    "h2_pinky_x","h2_pinky_y"
]

def vetorize(video_path, video_id=VIDEO_ID, show=False):
    cap = cv2.VideoCapture(video_path)
    data = []

    with mp_pose.Pose(0.5, 0.5) as pose, mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_res = pose.process(rgb)
            hand_res = hands.process(rgb)

            def pt(lm, i):
                return int(lm[i].x * w), int(lm[i].y * h)

            ls = le = lw = rs = re = rw = (-1, -1)

            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                ls, le, lw = pt(lm, 11), pt(lm, 13), pt(lm, 15)
                rs, re, rw = pt(lm, 12), pt(lm, 14), pt(lm, 16)

            h1 = [(-1, -1)] * 5
            h2 = [(-1, -1)] * 5

            if hand_res.multi_hand_landmarks:
                for i, hand in enumerate(hand_res.multi_hand_landmarks[:2]):
                    tips = [(int(hand.landmark[t].x * w), int(hand.landmark[t].y * h)) for t in [4, 8, 12, 16, 20]]
                    if i == 0: h1 = tips
                    else: h2 = tips

            row = [
                video_id,
                *ls, *le, *lw,
                *rs, *re, *rw,
                *(c for p in h1 for c in p),
                *(c for p in h2 for c in p)
            ]

            data.append(row)

            if show:
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    df = pd.DataFrame(data, columns=columns)
    return df