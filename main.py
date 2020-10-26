import os
import sys
import time
import pickle
import cv2
import numpy as np
import mbfn
import retinaface

if __name__ == "__main__":
    allf = pickle.load(open("features", 'rb'))
    print("database has:", len(allf), "persons")

    if len(sys.argv) != 2:
        print("Usage: %s [imagepath]\n"%(sys.argv[0]))
        sys.exit(0)
   
    mfn = mbfn.MobileFaceNetV3()

    imagepath = sys.argv[1]
    m = cv2.imread(imagepath)
    net =  retinaface.RetinaFace()
    s = time.time()
    faceobjects = net(m)
    lm = [[p.x,p.y] for p in faceobjects[0].landmark]
    lm = np.array(lm)
    features_a = mfn.extract(m, lm)
    print("used", time.time()-s, "s")
    for x in allf:
        features_b = x['features']
        similar = mbfn.return_similarity(features_a, features_b)
        #print("with: ", x['file'], similar)
        x['similar']  = similar

    result = sorted(allf, key = lambda kv:(kv['similar'], kv['file']), reverse=True)     
    most = 10
    for x in result:
        print(x['file'], x['similar'])
        most -= 1
        if most <= 0:
            break
