# %%
import cv2
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import torch

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
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# %%
class App:
    def __init__(self, img, model, target=None):
        self.root = Tk()
        self.model = model
        self.target = target
        self.ori_img = img.resize((224, 224))
        self.alpha = 0.1
        self.image = ImageTk.PhotoImage(self.ori_img)
        self.label = Label(self.root, image=self.image)
        self.button_pgd = Button(self.root, text="PGD_attack", command=self.pgd)
        self.slider = Scale(self.root, from_=0, to=0.5, resolution=0.01,
                                orient="horizontal", command=self.updateAlpha)

        self.target_choose = IntVar()
        self.radio_notarget = Radiobutton(self.root, text="No Attack", variable=self.target_choose, value=0, command=self.reset)
        self.radio_target = Radiobutton(self.root, text="Target Attack", variable=self.target_choose, value=1, command=self.pgd)
        self.radio_non_target = Radiobutton(self.root, text="Non-Target Attack", variable=self.target_choose, value=2, command=self.pgd)

        #self.slider.set(0.1)
        self.slider.grid(row=0, column=1)
        #self.button_pgd.grid(row=1, column=1)
        self.label.grid(row=0, column=0)
        self.radio_notarget.grid(row=2, column=0)
        self.radio_target.grid(row=2, column=1)
        self.radio_non_target.grid(row=2, column=2)

        self.root.mainloop()

    def pgd(self):
        model = self.model
        model.eval()
        X = transforms_test(self.ori_img).unsqueeze(0).to(device)
        y = model(X).argmax(1)
        epsilon, alpha, num_iter = self.alpha, self.alpha, 1

        delta = torch.rand_like(X, requires_grad=True)
        #set delta to be in the range of perturbation
        delta.data = delta.data * 2 * epsilon - epsilon

        for t in range(num_iter):
            
            yd = model(X + delta)
            loss = nn.CrossEntropyLoss()(yd, y)
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()

        #tensor to image
        
        new_img = (X + delta).squeeze(0).cpu().detach().numpy()
        new_img = new_img.transpose(1,2,0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        new_img = std * new_img + mean
        new_img = np.clip(new_img, 0, 1)
        new_img = Image.fromarray((new_img*255).astype(np.uint8))

        self.image = ImageTk.PhotoImage(new_img)
        self.label.configure(image=self.image)

         
    def updateAlpha(self, event):

        if self.target_choose.get() != 0:
            self.alpha = self.slider.get()
            self.pgd()

    def reset(self):
        self.image = ImageTk.PhotoImage(self.ori_img)
        self.label.configure(image=self.image)
        



# %%
img = Image.open("./CelebA_HQ_facial_identity_dataset/test/Xiang/0000.jpg")
model = torch.load("models/model_310_plus_max_pro_ultra.pt", map_location=device)

# %%
App(img, model, 307)


