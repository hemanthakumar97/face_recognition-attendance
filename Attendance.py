import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime,date
import imutils
import keyboard
# from PIL import ImageGrab
today = date.today()
date_Y = today.strftime("%Y")
date_M = today.strftime("%b")
date_D = today.strftime("%d-%b-%Y")

if not os.path.isdir(f'Attendance/{date_Y}'):
    os.mkdir(f'Attendance/{date_Y}')
if not os.path.isdir(f'Attendance/{date_Y}/{date_M}'):
    os.mkdir(f'Attendance/{date_Y}/{date_M}')

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name):
    attendance_file = f'Attendance/{date_Y}/{date_M}/{date_D}.csv'
    if not os.path.isfile(attendance_file):
        attendance_file = open(attendance_file, 'w+')
        attendance_file.close()
    with open( f'Attendance/{date_Y}/{date_M}/{date_D}.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
encodeListKnown = findEncodings(images)
print("[INFO] Encoding Complete")
print("[INFO] starting video stream...")
 
url = 'http://192.168.43.1:8080/video'
cap = cv2.VideoCapture(url)
 
while True:
    success, img = cap.read()
    img = imutils.resize(img, width=480)
    #img = captureScreen()
    # imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].title()
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
 
    cv2.imshow('Webcam',img)
    key = cv2.waitKey(10) & 0xFF

	# if the 'q' or 'ESC' key was pressed, break from the loop
    if keyboard.is_pressed('Esc') or key==ord('q'):
        print("[INFO] Program execution terminated by user")
        break

cap.release()
cv2.destroyAllWindows()
