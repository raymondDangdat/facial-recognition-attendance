import cv2
import numpy as np
import face_recognition
import os
from datetime import  datetime

path = 'images'
images = []
classNames = []
myList = os.listdir(path)

print(myList)

for cl in myList:
    currentImage = cv2.imread(f'{path}/{cl}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEndcodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open("attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')
        print(myDataList)

markAttendance('Dangdat')


encodeListKnown = findEndcodings(images)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgs)
    encodeCurrent = face_recognition.face_encodings(imgs, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrent, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches [matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0 ), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)





print('Encoding Complete')



# faceLoc = face_recognition.face_locations(ib)[0]
# encodeIb = face_recognition.face_encodings(ib)[0]
# cv2.rectangle(ib, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 200), 2)
#
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 200), 2)
#
# result = face_recognition.compare_faces([encodeIb], encodeTest)
# faceDis = face_recognition.face_distance([encodeIb], encodeTest)
# print(result, faceDis)
# cv2.putText(imgTest, f'{result} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

