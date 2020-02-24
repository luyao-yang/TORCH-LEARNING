from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models,transforms
import matplotlib.pyplot as plt
import os
import copy
import time



plt.ion()

#preprocess the data

data_transforms = {
        'train':transforms.Compose([
           transforms.RandomResizedCrop(224),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        }

data_dir='data/hymenoptera_data'

datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir,"train"),
                        transform=data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir,"val"),
                transform=data_transforms['val'])}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x],shuffle=True,
                                            batch_size=4,num_workers=4)
                for x in ['train','val']}

dataset_sizes= {x:len(datasets[x]) for x in ['train','val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model,criterion,optimizer,scheduler,num_epochs):
    if __name__ == '__main__':
        for epoch in range(num_epochs):
            print('Epoch{}/{}'.format(epoch,num_epochs-1))
            print('-'*10)

            for phase in ['train','val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0


                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    #torch.set_grad_enabled=torch.no_grad
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        #return the largest element and its index in each row
                        _,preds = torch.max(outputs,dim=1)
                        loss = criterion(outputs,labels)
                    
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                #each training epoch need update the learning rate
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss/dataset_sizes[phase]
                epoch_acc = running_corrects.double()/dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
                    

        print("Finished Training!")

#model.load_state_dict(torch.load('resnet18-5c106cde.pth'))
model = models.resnet18(pretrained=True)

#find the number of channels before the full-connected layer
num_firs = model.fc.in_features
model.fc = nn.Linear(num_firs,2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

model_fit = train_model(model,criterion,optimizer,exp_lr_scheduler,num_epochs=25)
