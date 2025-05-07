import cv2
import numpy as np
import face_recognition as fr

imgElon = fr.load_image_file('ImagesBasic/elon.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = fr.load_image_file('ImagesBasic/ElonTest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# detect face locations and encodings
faceLoc = fr.face_locations(imgElon)[0]
faceEnc = fr.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = fr.face_locations(imgTest)[0]
faceEncTest = fr.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

result = fr.compare_faces([faceEnc], faceEncTest)
faceDis = fr.face_distance([faceEnc], faceEncTest)
print(result, faceDis)
cv2.putText(imgTest, f'{result} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)





cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
