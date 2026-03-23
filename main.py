import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    h, w, _ = frame.shape

    # ===== POSE (ARMS) =====
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        # Left arm
        ls, le, lw = pt(11), pt(13), pt(15)
        cv2.line(frame, ls, le, (0,255,0), 3)
        cv2.line(frame, le, lw, (0,255,0), 3)

        # Right arm
        rs, re, rw = pt(12), pt(14), pt(16)
        cv2.line(frame, rs, re, (255,0,0), 3)
        cv2.line(frame, re, rw, (255,0,0), 3)

    # ===== HANDS (FINGERS) =====
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Highlight fingertips
            tips = [4, 8, 12, 16, 20]
            for tip in tips:
                x = int(hand_landmarks.landmark[tip].x * w)
                y = int(hand_landmarks.landmark[tip].y * h)
                cv2.circle(frame, (x, y), 8, (0,255,255), -1)

    cv2.imshow("Arms + Hands", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()