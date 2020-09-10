import dlib
import sys
import cv2

img = cv2.imread(sys.argv[1])
img_rd = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
faces = detector(img_rd, 0)
print (len(faces)):
shape = predictor(img_rd, faces[0])

for i in range(68):
    print(shape.part[i].x, shape.part[i].y)

