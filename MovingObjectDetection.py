import cv2
import numpy as np

cap = cv2.VideoCapture(0)

bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25,
    detectShadows=False
)
# bg_sub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fg_mask = bg_sub.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     if cv2.contourArea(contour) < 1000:
    #         continue

    for c in contours:
        if cv2.contourArea(c) > 1000:
            valid_contours = [c]
    if valid_contours:

        merged_contour = np.vstack(valid_contours)

        hull = cv2.convexHull(merged_contour)

        x, y, w, h = cv2.boundingRect(hull)

        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Motion Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()