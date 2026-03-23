import cv2
import mediapipe as mp
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

# Open a file to save positions
with open("arm_hand_positions.txt", "w") as f:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Stop after 3 seconds
        if time.time() - start_time >= duration:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(rgb)
        hand_results = hands.process(rgb)

        h, w, _ = frame.shape

        # Arms
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark

            def pt(i):
                return int(lm[i].x * w), int(lm[i].y * h)

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

            # Write arm positions
            f.write(f"Frame {frame_count} - Arms:\n")
            f.write(f"  Left: Shoulder {left_shoulder}, Elbow {left_elbow}, Wrist {left_wrist}\n")
            f.write(f"  Right: Shoulder {right_shoulder}, Elbow {right_elbow}, Wrist {right_wrist}\n")

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

                # Write fingertips
                f.write(f"Frame {frame_count} - Hand {hand_index + 1} fingertips: {fingertips}\n")

        cv2.imshow("Arms + Hands Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break  # still allow ESC to quit early

cap.release()
cv2.destroyAllWindows()