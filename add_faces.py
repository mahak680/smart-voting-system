import cv2
import pickle
import numpy as np
import os

if not os.path.exists('data/'):
    os.makedirs('data/')

video = cv2.VideoCapture(0)    # tostart your pc's acamera ##here 0 means 1 camera , 1 means two camera 
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_data = [] #empty array which is going to store images

i=0 #initially the loop is going to start from 0

# name = input("enter your name: ")
aadhar = input("enter your adhaar number: ")

totalFrames = 51 #how many images i need to cliock 
captureAfterFrames =  2                 #in 51 frames i want to keep a pause of 2 frames

while True:          #while True means cliking the images 
    ret, frame = video.read()           #start the video and the video first of all
    gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #convert that frame into gray, image is converted into grayscale
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data)<=totalFrames and i%captureAfterFrames == 0:
            faces_data.append(resized_img)
        i = i + 1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= totalFrames:
        break

video.release()
cv2.destroyAllWindows()

# print("length is faces data is :",len(faces_data))
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape((totalFrames, -1))
print(len(faces_data))

if 'aadharN.pkl' not in os.listdir('data/'):
    aadharN = [aadhar]*totalFrames
    with open('data/aadharN.pkl', 'wb') as f:
        pickle.dump(aadharN, f)
else:
    with open('data/aadharN.pkl', 'rb') as f:
        aadharN = pickle.load(f)
    aadharN = aadharN + [aadhar]*totalFrames
    with open('data/aadharN.pkl', 'wb') as f:
        pickle.dump(aadharN, f)         

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)   
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis = 0)
    with open ('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)             