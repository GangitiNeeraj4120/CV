import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
my_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = my_face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_W, _ = frame.shape
    if landmark_points:
        land_marks = landmark_points[0].landmark
        for idl, landmark in enumerate(land_marks[474:478]):
            x = int(landmark.x * frame_W)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if idl == 1:
                screen_x = int(landmark.x *screen_w)
                screen_y = int(landmark.y * screen_h)
                pyautogui.moveTo(screen_x, screen_y)

            left = [land_marks[145], land_marks[159]]
            for landmark in left:
                x = int(landmark.x*frame_W)
                y = int(landmark.y*frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
            if left[0].y - left[1].y < 0.004:
                pyautogui.click()


    cv2.imshow('EyeControlled mouse', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()