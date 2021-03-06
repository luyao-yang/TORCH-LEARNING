from __future__ import print_function,division
import torch
import os
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

#print(landmarks_frame)


n = 65 #take the 65th line image's name 
img_name = landmarks_frame.iloc[n,0] #the name
landmarks = landmarks_frame.iloc[n, 1:].values #change the image's landmark to a matrix
landmarks = landmarks.astype('float').reshape(-1, 2) #rehape the matrix

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1],s=50,marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),landmarks)

#must before the plt.show()
plt.ioff()# this function shows the according image
plt.show()


