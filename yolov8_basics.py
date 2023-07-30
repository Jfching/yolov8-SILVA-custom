from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  

# predict on an image
#detection_output = model.predict(source="inference/images/armor1.JPG", conf=0.25, save=True, show=True) 
detection_output = model.predict(source="1", conf=0.25, save=True, show=True) 

#show = true shows the camera GUI when code is executed

# Display tensor array
print(detection_output)

#test

# Display numpy array
print(detection_output[0].numpy())

