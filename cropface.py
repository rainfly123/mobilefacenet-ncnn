#!/usr/bin/env python
import sys
import cv2
import numpy as np
import time
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

if __name__ == '__main__':
        img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
        faces = detector(img, 0)
        if len(faces) == 1:
            for d in faces:
                height = (d.bottom() - d.top())
                width = (d.right() - d.left())
                img_blank = np.zeros((height, width, 3), np.uint8)
                for ii in range(height):
                    for jj in range(width):
                        img_blank[ii][jj] = img[d.top() + ii][d.left() + jj]
                cv2.imshow("d", img_blank)
                cv2.imwrite("abc.jpg", img_blank)
                cv2.waitKey(0)

    
