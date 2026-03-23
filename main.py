import cv2
import mediapipe as mp

# MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize models
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# Open video file
video_path = "videos/Armenian_Sign_Language_ArSL_բարև_ձեզ_hello!_1.webm"  # <-- replace with your video path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process models
    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    h, w, _ = frame.shape

    # ===== POSE (ARMS) =====
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        # Left arm
        left_shoulder, left_elbow, left_wrist = pt(11), pt(13), pt(15)
        cv2.line(frame, left_shoulder, left_elbow, (0, 255, 0), 3)
        cv2.line(frame, left_elbow, left_wrist, (0, 255, 0), 3)

        # Right arm
        right_shoulder, right_elbow, right_wrist = pt(12), pt(14), pt(16)
        cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 3)
        cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 3)

        # Draw joints
        for point in [left_shoulder, left_elbow, left_wrist,
                      right_shoulder, right_elbow, right_wrist]:
            cv2.circle(frame, point, 8, (0, 0, 255), -1)

    # ===== HANDS (FINGERS) =====
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Highlight fingertips
            fingertip_ids = [4, 8, 12, 16, 20]
            for tip in fingertip_ids:
                x = int(hand_landmarks.landmark[tip].x * w)
                y = int(hand_landmarks.landmark[tip].y * h)
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

    # Show frame
    cv2.imshow("Arms + Hands Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()