import cv2 as cv
import numpy as np
import mediapipe as mp

from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
from utils import * 
import time 

import json

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object

#non-targeted attack
def pgd(model, X, y, epsilon, alpha, num_iter):
    delta = torch.rand_like(X, requires_grad=True)
    #set delta to be in the range of perturbation
    delta.data = delta.data * 2 * epsilon - epsilon

    for t in range(num_iter):
        
        yd = model(X + delta)
        loss = nn.CrossEntropyLoss()(yd, y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()

        
    return delta.detach()

#targeted attack
def pgd_target(model, X, y_target, epsilon, alpha, num_iter):

    delta = torch.rand_like(X, requires_grad=True)
    #set delta to be in the range of perturbation
    delta.data = delta.data * 2 * epsilon - epsilon

    #get fake labels as second highest probability
    #y_fake = torch.argsort(model(X), dim=1)[:, -2]

    for t in range(num_iter):

        yd = model(X + delta)
        loss = nn.CrossEntropyLoss()(yd, y_target)
        loss.backward()
        delta.data = (delta - alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
        
    return delta.detach()


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
# model = models.resnet18(pretrained=True)
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 310) # multi-class classification (num_of_class == 310)
# model.load_state_dict(torch.load('models/model_310.pth', map_location=device))

model = torch.load('models/model_310_plus_max_pro_ultra.pt', map_location=device)

# Load class names
f = open('models/class_names_310.json' , 'r')
classNames = json.load(f)
f.close()

#cpature the camera
cap = cv.VideoCapture(1)

#for each frame
while True:
    success, frame = cap.read()
    if not success:
        break
    
    
    x, y, c = frame.shape
    # Convert the BGR frame to RGB before processing.
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Detect the faces 
    result = face_detection.process(frame)
    
    
    className = ''

    # post process the result
    if result.detections:
        for detection in result.detections:
            #get the bounding box
            bbox = detection.location_data.relative_bounding_box
            #get the center of the bounding box
            center_x = bbox.xmin + bbox.width / 2 # x coordinate of the center
            center_y = bbox.ymin + bbox.height / 2  # y coordinate of the center

            center_x = int(center_x * frame.shape[1]) # convert to pixel
            center_y = int(center_y * frame.shape[0]) # convert to pixel

            size = int(max(bbox.width * frame.shape[1], bbox.height * frame.shape[0])*1) # convert to pixel
            size = int(size/2) + 10 # add some padding
            image = frame[center_y-size-5:center_y+size-5, center_x-size:center_x+size] # crop the face
            face_shape = image.shape 
        
            if image.shape[0] == 0 or image.shape[1] == 0:
                continue
            
            # crop the face and classify it
            image = Image.fromarray(image)
            image = transforms_test(image) # convert to tensor
            image = image.unsqueeze(0) # add a batch dimension
            image = image.to(device) # send to device
            model.eval() # set model to evaluation mode
            outputs = model(image) # get the output
            _, preds = torch.max(outputs, 1) # get the predicted class
            likelihood = outputs.softmax(1).max(1)[0].item() # get the likelihood of the prediction
            
            
            
            
            
            
            
            # pgd attack: non-targeted and targeted
            # un-comment the one you want to use
            # delta = pgd(model, image, preds, 0.1, 0.05, 1)
            delta = pgd_target(model, image, torch.Tensor([64]).type(torch.LongTensor), 0.5, 0.1, 3)
            
            
            
            
            
            
            # get the prediction of the perturbed image
            img_pgd = image + delta
            outputs = model(img_pgd) # get the output
            _, pgd_preds = torch.max(outputs, 1) # get the predicted class
            pgd_likelihood = outputs.softmax(1).max(1)[0].item() # get the likelihood of the prediction
            
            className = classNames[preds.item()]
            pgd_className = classNames[pgd_preds.item()]
            
            
            # Display the predictions and the bounding box
            cv.rectangle(frame, (center_x-size, center_y-size-5), (center_x+size, center_y+size-5), (0, 255, 0), 2)
            cv.putText(frame, 'True: ' +  className + ' ' + str(likelihood)[:4], (center_x-size, center_y-size-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(frame, 'Encrypted: ' +  pgd_className + ' ' + str(pgd_likelihood)[:4], (center_x-size, center_y-size-35), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # #replace the face with the encrypted face
            img_pgd = img_pgd[0].permute(1,2,0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_pgd = std * img_pgd + mean
            img_pgd *= 255.0
            #rescale the image to the range of [0, 255]
            img_pgd = np.clip(img_pgd, 0, 255)
            img_pgd = img_pgd.astype(np.uint8)
            # resize the encrypted face to the original size
            img_pgd = cv.resize(img_pgd, (face_shape[1], face_shape[0])) 


            frame[center_y-size-5:center_y+size-5, center_x-size:center_x+size] = img_pgd # replace the face with the encrypted face
    else:
        cv.putText(frame, 'No face detected', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Display the predicted gesture and the bounding box
    
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow("Face Encryption Demo", frame)
    # out.write(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv.destroyAllWindows()