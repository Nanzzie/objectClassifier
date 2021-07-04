#OBJECT_CLASSIFICATION
import cv2
from cv2 import dnn_DetectionModel


configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(configPath ,weightsPath)
classNames = [ ]  #empty list of python 
classFiles = 'coco.names'
with open(classFiles,'rt') as fpt:
     classNames = fpt.read().rstrip('\n').split('\n')
      #classNames.append(fpt.read())
model.setInputSize(320,320)
model.setInputScale(1.0/ 127.5)  #255/2=127.5
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)

cap.set(3,140)
cap.set(4,180)
cap.set(10,70)

while True :
    ret , img = cap.read( )
    classIds , confidence , bbox = model.detect( img , confThreshold = 0.55 )
print(classIds)

if len(classIds) != 0:
    for classId, conf,box in zip(classIds.flatten(), confidence.flatten(), bbox):
        if(classId<=91):
            cv2.rectangle ( img , box , color = (0 , 255 , 0) , thickness = 2 )
            cv2.putText ( img , classNames [ classId - 1 ] , (box [ 0 ] + 10 , box [ 1 ] + 30) ,
                          cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 255 , 0) , 2 )
           # cv2.putText ( img , str ( round ( confidence * 100 , 2 ) ) , (box [ 0 ] + 200 , box [ 1 ] + 30) ,
                    #     cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 255 , 0) , 2 )


        cv2.imshow( 'img' , img )
        if cv2.waitKey( 1 ) & OxFF == ord( 'q' ) :
         break

cap.release( )
cv2.destroyAllWindows( )
