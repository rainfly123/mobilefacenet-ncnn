import dlib
import sys
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def landmarks(img_rd):
    faces = detector(img_rd, 0)
    print(len(faces))
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
    import time
    s = time.time()
    img_rd = cv2.imread(sys.argv[1])
    total, loc = landmarks(img_rd)
    print(total, loc)
    print(time.time()-s)

