import cv2
import os
import supervision as sv
from ultralytics import YOLO


model = YOLO("best.pt")

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")

img_counter = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    annotated_image = bounding_box_annotator.annotate(
    scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)   
    
    cv2.imshow('frame', annotated_image)
    
    k = cv2.waitKey(1)
    
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
        
cap.release()
cv2.destroyAllWindows()