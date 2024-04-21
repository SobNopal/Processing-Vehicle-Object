from ultralytics import YOLO
import cv2, numpy as np, time

modelPath="D:/NAUFAL/File Naufal/Semester 6/Pengolahan Citra Digital/Naufal/model/train/weights"
model=YOLO(modelPath)#model tertraining coco dataset

cam=cv2.VideoCapture(0)
imgSize=640

def new_func(data):
    Bbox=data.boxes.data.tolist()[0]
    return Bbox

while True:
    ret, img=cam.read()
    if ret:
        now=time.time()
        img=cv2.resize(img, (640,320))
        result=model.predict(img, stream=True, imgsz=640, verbose=False, conf=0.5)
        for data in (result):
            for i in range (len(data)):
                Bbox=data.boxes.data.tolist()[i]
                Class=data.names
                x1, y1, x2, y2, score, clsid=Bbox
                p1=np.int16((x1,y1))
                p2=np.int16((x2,y2))
                mX=np.int16(x1+((x2-x1)/2))
                mY=np.int16(y1+((y2-y1)/2))
                objctClass=Class[clsid]
                cv2.rectangle(img, p1, p2, (0,0,255),2)
                text=objctClass+""+str(round(score,2))
                cv2.putText(img, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                cv2.putText(img, str(mX)+","+str(mY), (mX, mY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    elapsedTime=time.time()-now
    fps=round(1/elapsedTime, 2)
    cv2.putText(img, "FPS: "+str(fps), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,20), 2)
    cv2.imshow("Detection", img)
    cv2.waitKey(1)     