
import cv2
import dlib
from math import hypot

# create default face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX
mouth_points = [48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
#입모양 비슷한 특징점끼리 이어주기
r_eye_points = [42, 43, 44, 45, 46, 47]
#오른쪽 눈 비슷한 특징점끼리 이어주기
l_eye_poits = [36, 37, 38, 39, 40, 41]
#왼쪽 눈 비슷한 특징점끼리 이어주기
count_mouth_open = 0

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
#중점 지점해주기

def get_mouth_pen_ratio(mouth_points, facial_landmarks):
    left_point = (facial_landmarks.part(
        mouth_points[12]).x, facial_landmarks.part(mouth_points[12]).y)
    right_point = (facial_landmarks.part(
        mouth_points[16]).x, facial_landmarks.part(mouth_points[16]).y)
    center_top = midpoint(facial_landmarks.part(
        mouth_points[13]), facial_landmarks.part(mouth_points[14]))
    center_bottom = midpoint(facial_landmarks.part(
        mouth_points[19]), facial_landmarks.part(mouth_points[18]))
#입 모양을 감지할 수 있게 포인트별로 범위를 정해줌

    hor_line = cv2.line(image, left_point, right_point, (0, 255, 0), 2)   # 가로
    ver_line = cv2.line(image, center_top, center_bottom, (0, 255, 0), 2) # 세로
#종횡비를 구해서 움직임을 감지

    hor_line_lenght = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    if ver_line_lenght != 0:
        ratio = hor_line_lenght / ver_line_lenght
    else:
        ratio = 60
    return ratio


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(image, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(image, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    # 눈의 경우 ratio가 높아지면 눈을 감은 상태
    return ratio

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True :
    _, image = capture.read()

    # convert frame to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        mouths = get_mouth_pen_ratio(
            mouth_points, landmarks)

        if mouths <= 5.0:
            count_mouth_open += 1

        left_eye_ratio = get_blinking_ratio(
            l_eye_poits, landmarks)
        right_eye_ratio = get_blinking_ratio(
            r_eye_points, landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blinking_ratio >= 6.0:
            cv2.putText(image, "blinking", (50, 50), font, 2, (255, 0, 0))
            print("blinking")

    cv2.putText(image, "Mouth open: " + str(count_mouth_open),
                (50, 50), font, 2, (255, 0, 0))
    print("Mouth open: " + str(count_mouth_open))

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break