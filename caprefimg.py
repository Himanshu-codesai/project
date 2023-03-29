import cv2 as cv
cap = cv.VideoCapture(0)
cap.set(3,1080)
cap.set(4,720)
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
classNames = []
classFile = 'classes.txt'

detector = HandDetector(detectionCon=0.8,maxHands=1)
x = [300,250,200,150,130,115,100,80,70,66,50]
y = [12,14,19,23,27,34,38,46,52,62,70]
coff = np.polyfit(x,y,2)


with open(classFile,'r') as f:
    classNames =[cname.strip() for cname in f.readlines()]
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
fonts = cv.FONT_HERSHEY_COMPLEX

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    classes, scores, boxes = model.detect(img, 0.4, 0.4)
    data_list = []


    for (classid, score, box) in zip(classes, scores, boxes):


        # label = "%s : %f" % (classNames[classid[0]], score)
        # label = "%s : %f" % (classNames[classid[0]], score)


        cv.rectangle(img, box, (0,255,0), 2)
        cv.putText(img,classNames[classid], (box[0], box[1]-10), fonts, 0.5, (0,255,0), 2)

        if classid == 0:
            # person class id
            data_list.append([classNames[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 67:
            data_list.append([classNames[classid], box[2], (box[0], box[1] - 2)])
        print(data_list)
        # return list

    if hands:
        lmList = hands[0]['lmList']
        x,y,w,h = hands[0]['bbox']
        x1, y1,z1 = lmList[5]
        x2, y2,z2 = lmList[17]
        distance = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))
        A,B,C = coff
        distanceCM = A*distance**2+B*distance+C
        print(distanceCM)
        cvzone.putTextRect(img,f'{int(distanceCM)}cm',(x,y))

    cv.imshow('sf', img)
    cv.waitKey(1)



