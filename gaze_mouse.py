import cv2
import mediapipe as mp
import numpy as np
import pyautogui as py
import time

py.FAILSAFE = True
MOVE_SPEED = 25
GAZE_HOLD_TIME = 4.0

LEFT_THRESHOLD = 0.35
RIGHT_THRESHOLD = 0.65
UP_THRESHOLD = 0.25

EAR_BLINK_THRESH = 0.18
BLINK_MAX_DURATION = 0.25
BLINK_GROUP_TIME = 0.8

EAR_THRESH = 0.20
BLINK_MAX_TIME = 0.25
BLINK_WINDOW = 0.9

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

ear_low_start = None
blink_counter = 0
last_blink_time = 0
ear_closed_time = None
eye_closed = False

blink_start_time = 0
lasr_action_time = time.time()
gaze_hold_start = None

def eye_aspect_ratio(eye_points):
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return (vertical1 + vertical2) / (2.0*horizontal) #Detects eye blinks

def get_gaze_ratio(eye_points, iris_points):
    eye_left_x = eye_points[0][0]
    eye_right_x = eye_points[3][0]
    iris_x = iris_points[0]
    return (iris_x - eye_left_x) / (eye_right_x - eye_left_x) #Gaze_ratio = iris distance from left/ eye width

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) #mirror imaging

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Converts to RGB cuz Mediapipe trained on RGB images
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks is not None:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        left_eye_pts = np.array(
            [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in left_eye]
        )
        right_eye_pts = np.array(
            [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in right_eye]
        )

        left_iris = landmarks[468]
        iris_x = int(left_iris.x*w)
        iris_y = int(left_iris.y*h)

        for (x, y) in left_eye_pts:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        for (x, y) in right_eye_pts:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.circle(frame, (iris_x, iris_y), 3, (0, 0, 255), -1)

        #Gaze movemont
        gaze_ratio = get_gaze_ratio(left_eye_pts, (iris_x, iris_y))

        ear = (eye_aspect_ratio(left_eye_pts)+eye_aspect_ratio(right_eye_pts))/2

        current_time = time.time()

        moved = False

        if gaze_ratio < LEFT_THRESHOLD:
            py.moveRel(-MOVE_SPEED, 0)
            moved = True
        elif gaze_ratio > RIGHT_THRESHOLD:
            py.moveRel(MOVE_SPEED, 0)
            moved = True
        else:
            top_y = left_eye_pts[1][1]
            bottom_y = left_eye_pts[4][1]
            iris_pos = (iris_y - top_y)/(bottom_y - top_y)

            if iris_pos < UP_THRESHOLD:
                py.moveRel(0, -MOVE_SPEED)
                moved = True
            print(f"iris_pos: {iris_pos:.2f}")


        #Center Gaze
        if not moved:
            if gaze_hold_start is None:
                gaze_hold_start = time.time()
            elif time.time() - gaze_hold_start >= GAZE_HOLD_TIME:
                py.click()
                gaze_hold_start = None
        else:
            gaze_hold_start = None


        if ear < EAR_THRESH:
            if ear_closed_time is None:
                ear_closed_time = current_time
        else:
            if ear_closed_time is not None:
                duration = current_time - ear_closed_time

                if duration <= BLINK_MAX_TIME:
                    if current_time - last_blink_time <= BLINK_WINDOW:
                        blink_counter += 1
                    else:
                        blink_counter = 1

                    last_blink_time = current_time

                ear_closed_time = None

        if blink_counter == 2:
            py.click()
            print("Left Click")
            blink_counter = 0
        elif blink_counter == 3:
            py.rightClick()
            print("Right Click")
            blink_counter = 0

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Gaze Controlled Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()