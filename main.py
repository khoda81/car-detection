import cv2
from ultralytics import YOLO

model=YOLO('export v1\yolov8s.pt')

cv2.namedWindow('RGB')
# Open the video file
video_path = "dataset/onewayup.mp4"
cap = cv2.VideoCapture(video_path)


while True:   
    ret,frame = cap.read()
    if not ret:
        break
    
    frame=cv2.resize(frame,(1020,500))
   

    results=model.track(frame,persist=True)
     
    for x1,y1,x2,y2,id,conf,cls in results[0].boxes.data.cuda():
        cls=model.names[int(cls)]
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
        cv2.putText(frame,f"{cls}:{id}",(int(x1),int(y1)),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

