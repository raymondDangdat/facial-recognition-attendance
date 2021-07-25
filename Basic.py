import cv2
import numpy as np
import face_recognition

ib = face_recognition.load_image_file('images/dangdat.jpg')
ib = cv2.cvtColor(ib, cv2.COLOR_BGR2RGB)


imgTest = face_recognition.load_image_file('images/dangdat_test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(ib)[0]
encodeIb = face_recognition.face_encodings(ib)[0]
cv2.rectangle(ib, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 200), 2)


faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 200), 2)

result = face_recognition.compare_faces([encodeIb], encodeTest)
faceDis = face_recognition.face_distance([encodeIb], encodeTest)
print(result, faceDis)
cv2.putText(imgTest, f'{result} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Ibrahim', ib)
cv2.imshow('Ibrahim Test', imgTest)
cv2.waitKey(0)
