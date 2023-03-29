import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)

# Distance constants
KNOWN_DISTANCE = 24  # INCHES
PERSON_WIDTH = 16.9  # INCHES
MOBILE_WIDTH = 2.95  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
classNames = []
classFile = 'classes.txt'

with open(classFile,'r') as f:
    classNames =[cname.strip() for cname in f.readlines()]
# with open("classes.txt", "r") as f:
#     class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
# yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
fonts = cv.FONT_HERSHEY_COMPLEX
#
# # model = cv.dnn_DetectionModel(yoloNet)
# # model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
#
# def focal_length_finder (measured_distance, real_width, width_in_rf):
#     focal_length = (width_in_rf * measured_distance) / real_width
#     return focal_length
# # distance finder function
# def distance_finder(focal_length, real_object_width, width_in_frmae):
#     distance = (real_object_width * focal_length) / width_in_frmae
#     return distance
#
# ref_person = cv.imread('ReferenceImages/image1.png')
# ref_mobile = cv.imread('ReferenceImages/image1.png')
#
# mobile_data = object_detector(ref_mobile)
# mobile_width_in_rf = mobile_data[1][1]
#
# person_data = object_detector(ref_person)
# person_width_in_rf = person_data[0][1]
#
# print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")
#
# # finding focal length
# focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
#
# focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
# cap = cv.VideoCapture(0)
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# Recoder = cv.VideoWriter('out.mp4', fourcc,8.0,(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))) )
#
# while True:
#     success, img = cap.read()
#     classes, scores, boxes = model.detect(img, 0.4, 0.4)
#     data_list = []
#
#
#     for (classid, score, box) in zip(classes, scores, boxes):
#
#
#         # label = "%s : %f" % (classNames[classid[0]], score)
#         # label = "%s : %f" % (classNames[classid[0]], score)
#
#
#         cv.rectangle(img, box, (0,255,0), 2)
#         cv.putText(img,classNames[classid], (box[0], box[1]-10), fonts, 0.5, (0,255,0), 2)
#
#         if classid == 0:
#             # person class id
#             data_list.append([classNames[classid], box[2], (box[0], box[1] - 2)])
#         elif classid == 67:
#             data_list.append([classNames[classid], box[2], (box[0], box[1] - 2)])
#         print(data_list)
#         # return list
#         data=data_list
#         for d in data:
#             if d[0] == 'person':
#                 distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
#                 x, y = d[2]
#             elif d[0] == 'cell phone':
#                 distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
#                 x, y = d[2]
#             cv.rectangle(img, (x, y - 3), (x + 150, y + 23), BLACK, -1)
#             cv.putText(img, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)
#
#         cv.imshow('frame', img)
#         Recoder.write(img)
#
#         key = cv.waitKey(1)
#         if key == ord('q'):
#             break
#     cv.destroyAllWindows()
#     Recoder.release()
#     cap.release()

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        # label = "%s : %f" % (classNames[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, classNames[classid], (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([classNames[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 67:
            data_list.append([classNames[classid], box[2], (box[0], box[1] - 2)])
        # return list
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


# reading the reference image from dir
ref_person = cv.imread('ReferenceImages/image1.png')
ref_mobile = cv.imread('ReferenceImages/image1.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
Recoder = cv.VideoWriter('out.mp4', fourcc, 8.0,
                         (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
while True:
    ret, frame = cap.read()
    data = object_detector(frame)
    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)
    Recoder.write(frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
Recoder.release()
cap.release()
