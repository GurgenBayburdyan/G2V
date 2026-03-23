import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Get arm points
        def get_point(id):
            return int(landmarks[id].x * w), int(landmarks[id].y * h)

        left_shoulder = get_point(11)
        left_elbow = get_point(13)
        left_wrist = get_point(15)

        right_shoulder = get_point(12)
        right_elbow = get_point(14)
        right_wrist = get_point(16)

        # Draw left arm
        cv2.line(frame, left_shoulder, left_elbow, (0,255,0), 3)
        cv2.line(frame, left_elbow, left_wrist, (0,255,0), 3)

        # Draw right arm
        cv2.line(frame, right_shoulder, right_elbow, (255,0,0), 3)
        cv2.line(frame, right_elbow, right_wrist, (255,0,0), 3)

        # Draw points
        for point in [left_shoulder, left_elbow, left_wrist,
                      right_shoulder, right_elbow, right_wrist]:
            cv2.circle(frame, point, 8, (0,0,255), -1)

    cv2.imshow("Arm Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()