import cv2
import dlib
from scipy.spatial import distance


# ฟังก์ชันคำนวณ Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# โหลดตัวตรวจจับใบหน้าและตัววัดจุด Landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# ดัชนีของตำแหน่งดวงตาซ้ายและขวา
(left_eye_start, left_eye_end) = (42, 48)
(right_eye_start, right_eye_end) = (36, 42)

# ค่าที่กำหนด EAR ต่ำกว่าแสดงว่าดวงตาหลับ
EAR_THRESHOLD = 0.25
# จำนวนเฟรมที่หลับตาติดต่อกัน
EAR_CONSEC_FRAMES = 30

# ตัวนับจำนวนเฟรมที่ดวงตาปิด
counter = 0
alarm_on = False

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # ดึงตำแหน่งดวงตาซ้ายและขวา
        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]

        # คำนวณ EAR สำหรับทั้งสองตา
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # ค่ากลาง EAR ของทั้งสองตา
        ear = (left_ear + right_ear) / 2.0

        # วาดรูปจุดที่ดวงตา
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # ถ้า EAR ต่ำกว่าค่ากำหนด ให้เพิ่มตัวนับเฟรม
        if ear < EAR_THRESHOLD:
            counter += 1

            # ถ้าเฟรมที่หลับตามากกว่าหรือเท่ากับจำนวนที่กำหนดให้แสดงเตือน
            if counter >= EAR_CONSEC_FRAMES:
                if not alarm_on:
                    print("เตือน: หลับตา!")
                    alarm_on = True

                cv2.putText(frame, "Do not sleep while drive!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            counter = 0
            alarm_on = False

    cv2.imshow("Drowsiness Detector", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
