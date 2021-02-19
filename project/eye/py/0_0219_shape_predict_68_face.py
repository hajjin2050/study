import numpy as np
import cv2
import dlib

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#얼굴의 각 구역의 포인트들을 구분.
JAULINE_POINTS = list(range(0,17))
RIGHT_EYEBROW_POINTS = list(range(17,22))
LEFT_EYEBROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42,48))
MOUTH_OUTLINE_POINTS = list(range(48,61))
MOUTH_INNER_POINTS = list(range(61, 68))

def detect(gray, frame):
    #일단 , 등록한 cascade classfier 를 이용해 얼굴찾기
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 5,minSize=(100, 100))

    #얼굴에서 랜드마크 찾기
    for(x, y, w, h) in faces:
        #open cv 로 dlib용 사각형 변환
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        #랜드마크 포인트들 지정
        landmarks = np.matrix([[p.x , p.y] for p in predictor(frame , dlib_rect).parts()])
        #원하는 포인트들을 넣는다
        landmarks_display - landmarks[0:60] #지금은 범위 전부
        #눈만 = landmarks_display = landmarks[RIGHT_EYE_POINTS , LEFT_EYE_POINTS]

        #포인트 출력
        for idx, point in enumertae(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness =-1)
    return frame