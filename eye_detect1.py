import numpy as np
import dlib
import cv2
import math

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)
        left_point1 = (landmarks.part(36).x, landmarks.part(36).y)
        right_point1 = (landmarks.part(39).x, landmarks.part(39).y)
        #center_top = midpoint(landmarks.part(37), landmarks.part(38))
        #center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        left_point2 = (landmarks.part(42).x, landmarks.part(42).y)
        right_point2 = (landmarks.part(45).x, landmarks.part(45).y)

        cv2.circle(frame, left_point1, 3, (0,0,255),2)
        cv2.circle(frame, right_point1, 3, (0,0,255),2)
        cv2.line(frame, left_point1, right_point1, (0, 255, 0), 2)

        cv2.circle(frame, left_point2, 3, (0,0,255),2)
        cv2.circle(frame, right_point2, 3, (0,0,255),2)
        cv2.line(frame, left_point2, right_point2, (0, 255, 0), 2)

        #cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        #lenv=math.sqrt((center_top[0] - center_bottom[0])**2 + (center_top[1] - center_bottom[1]) **2)
        lenh=math.sqrt((left_point1[0] - right_point1[0])**2 + (left_point1[1] - right_point1[1]) **2)

        #lennv = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        #lennh = math.hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

        #rat=lenh/lenv

        print(lenh)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
