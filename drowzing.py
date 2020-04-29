import numpy as np
import dlib
import cv2
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def compute_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)

        outer_lip_ratio = compute_blinking_ratio([48,49,50,51,52,53,54,55,56,57,58,59], landmarks)
        inner_lip_ratio = compute_blinking_ratio([60,61,62,63,64,65,66,67], landmarks)
        yawning_ratio = (outer_lip_ratio + inner_lip_ratio) / 2
        cv2.putText(frame, str(yawning_ratio), (30, 30), font, 2, (100, 100, 100))

        print(outer_lip_ratio,inner_lip_ratio,yawning_ratio)
        if yawning_ratio > 1.4:
            cv2.putText(frame, "yawning", (50, 150), font, 7, (0, 0, 255))


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
