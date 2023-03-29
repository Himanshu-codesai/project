import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
from cvzone.HandTrackingModule import HandDetector
import math
import cvzone
#
# detector = HandDetector(detectionCon=0.8,maxHands=1)
# x = [300,250,200,150,130,115,100,80,70,66]
# y = [12,14,19,23,27,34,38,46,52,62]
url = 'http://192.168.231.1/cam-hi.jpg'
im = None
classNames = []
classFile = 'coco.names'

with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightspath = 'frozen_inference_graph.pb'

net =cv2.dnn_DetectionModel(weightspath,configPath)
# coff = np.polyfit(x,y,2)

net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        cv2.imshow('live transmission', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)
        classIds, confidencess, bbox = net.detect(im, confThreshold=0.5)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confidencess.flatten(), bbox):
                cv2.rectangle(im, box, color=(0, 255, 0), thickness=3)
                cv2.putText(im, classNames[classId - 1].capitalize(), (box[0] + 10, box[1] + 30), cv2.FONT_ITALIC, 1,
                            (255, 255, 0), 2)
                cv2.putText(im, str(round(confidence * 100, 2)), (box[0] + 40, box[1] + 50), cv2.FONT_ITALIC, 0.5,
                            (255, 255, 0), 1)
        # out.write(im)
        # result = cv2.VideoWriter('filename.avi',
        #                          cv2.VideoWriter_fourcc(*'MJPG'),
        #                          10, size)
        cv2.imshow('detection', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

# def run3():
#     cv2.namedWindow("live distance", cv2.WINDOW_AUTOSIZE)
#     while True:
#         img_resp = urllib.request.urlopen(url)
#         imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         im = cv2.imdecode(imgnp, -1)
#         hands, img = detector.findHands(im)
#         if hands:
#             lmList = hands[0]['lmList']
#             x, y, w, h = hands[0]['bbox']
#
#             # print(lmList[5])
#             x1, y1, z1 = lmList[5]
#             x2, y2, z2 = lmList[17]
#             # print(x1)
#             distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
#             A, B, C = coff
#             distanceCM = A * distance ** 2 + B * distance + C
#             print(distanceCM)
#             # print(distance)
#             # print(distance)
#             cvzone.putTextRect(img, f'{int(distanceCM)}cm', (x, y))
#
#             # print(abs(x2-x1),distance)
#         cv2.imshow("distance", im)
#         key = cv2.waitKey(5)
#         if key == ord('q'):
#             break
#
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
        f1 = executer.submit(run1)
        f2 = executer.submit(run2)
        # f3 = executer.submit(run3)
