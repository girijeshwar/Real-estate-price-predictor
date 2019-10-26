import cv2
from datetime import datetime
import pandas as pd
#import times
a=1
times=[]
camera_port = 0
video = cv2.VideoCapture(camera_port , cv2.CAP_DSHOW)
first_frame = None
status_list=[None, None]
df=pd.DataFrame(columns=["Start","End"])
while True:
    a=a+1
    status=0
    check, frame = video.read()
    print(frame)
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    if first_frame is None:
        first_frame = gray
        continue
    delta_frame= cv2.absdiff(first_frame, gray)
    thresh_delta = cv2.threshold(delta_frame, 30, 255,cv2.THRESH_BINARY)[1]
    thresh_delta=cv2.dilate(thresh_delta,None, iterations=0)
    contours, hierachy= cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1
        (x,y,w,h)= cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y), (x+w, y+h),(0,255,0),3)
    status_list.append(status)
    status_list=status_list[-2:]

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now)

    cv2.imshow('video', thresh_delta)
    cv2.imshow('capturing', gray)
    cv2.imshow('delta', delta_frame)
    cv2.imshow('theshold', frame)
    key = cv2.waitKey(1)
    if key ==ord('q'):
        break
print(a)
print(status_list)
print(times)
for i in range(0, len(times),1):
    df=df.append({"start":times[i], "End":times[i+1]},ignore_index=True)
df=df.to_csv("times.csv")
video.release()
cv2.destroyAllWindows()