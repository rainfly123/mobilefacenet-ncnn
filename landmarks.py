import dlib
import sys
import cv2

def landmarks(img_rd):
    img_rd = cv2.resize(img_rd, (112, 112))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = detector(img_rd, 0)
    total  = len(faces)
    loc = list()
    for i in range(total) :
        shape = predictor(img_rd, faces[i])
        points = [
                 [shape.part(37).x + 2, shape.part(37).y + 2],
                 [shape.part(43).x + 2, shape.part(43).y + 2],
                 [shape.part(30).x, shape.part(30).y],
                 [shape.part(48).x + 2 , shape.part(48).y],
                 [shape.part(54).x + 2 , shape.part(54).y]]
        loc.append(points)
    return total, loc

if __name__ == "__main__":
    img_rd = cv2.imread(sys.argv[1])
    total, loc = landmarks(img_rd)
    print(total, loc)
