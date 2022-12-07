
import cv2 as cv
import numpy as np
import mediapipe as mp

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

import time
import os

import pandas as pd
from utils import * 

import json

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object


# initialize mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load the classification model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 310) # multi-class classification (num_of_class == 310)
model.load_state_dict(torch.load('models/model_310.pth', map_location=device))

# Load class names
f = open('models/class_names_310.json' , 'r')
classNames = json.load(f)
f.close()


cap = cv.VideoCapture(1)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    
    x, y, c = frame.shape
    # Convert the BGR frame to RGB before processing.
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Detect the hand landmarks. 
    result = face_detection.process(frame)
    
    
    className = ''

    
    # post process the result
    if result.detections:

        for detection in result.detections:
            #get the bounding box
            bbox = detection.location_data.relative_bounding_box
            #get the center of the bounding box
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2 

            center_x = int(center_x * frame.shape[1])
            center_y = int(center_y * frame.shape[0])

            size = int(max(bbox.width * frame.shape[1], bbox.height * frame.shape[0])*1)
            size = int(size/2) + 10
            image = frame[center_y-size-5:center_y+size-5, center_x-size:center_x+size]
        
        if image.shape[0] == 0 or image.shape[1] == 0:
            continue

        
        image = Image.fromarray(image)
        
        # # show the image
        # cv.imshow('image', image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # convert to tensor
        image = transforms_test(image) 
        

        # normalize the image
        #image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        image = image.unsqueeze(0)
        
        # # show the image
        # img = image.squeeze(0)
        # img = img.numpy()
        # img = np.transpose(img, (1, 2, 0))
        # img = Image.fromarray((img*255).astype(np.uint8))

        # #display image
        # img.show()

        
        
        # predict the class
        with torch.no_grad():
            output = model(image.to(device))
            _, predicted = torch.max(output, 1)
            className = classNames[predicted.item()]
    
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        # Display the predicted gesture and the bounding box
        cv.rectangle(frame, (center_x-size, center_y-size-5), (center_x+size, center_y+size-5), (0, 255, 0), 2)
        cv.putText(frame, className, (center_x-size, center_y-size-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.putText(frame, 'No face detected', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    cv.imshow("Face Encryption Demo", frame)
    # out.write(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv.destroyAllWindows()