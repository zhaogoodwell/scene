import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from scencedataset import Exampler
import torch.nn.functional as F
use_gpu = torch.cuda.is_available()



batch_size = 24*6
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


trainset_path_base  = "/home/ime/work/ai/dataset/ai_challenger_scene_train_20170904/"
trainset_image_path = os.path.join(trainset_path_base,'scene_train_images_20170904')
train_label_path    = trainset_path_base +"scene_train_annotations_20170904.json"

val_path_base   = "/home/ime/work/ai/dataset/ai_challenger_scene_validation_20170908/"
val_image_path  = os.path.join(val_path_base,"scene_validation_images_20170908")
val_label_path  = val_path_base +"scene_validation_annotations_20170908.json"

trainset = Exampler(label_path = train_label_path, image_path = trainset_image_path,transforms=data_transforms['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=32)

valset   = Exampler(label_path = val_label_path, image_path =val_image_path,transforms=data_transforms['val'])
valloader   = torch.utils.data.DataLoader(valset,batch_size=batch_size,shuffle=False,num_workers=32)
dataloder ={'train':trainloader,'val':valloader}
dataset_sizes ={'train':len(trainset),'val':len(valset)}

class Net(nn.Module):
    def __init__(self,num_class):
        super(Net,self).__init__()
        self.net = models.resnet152(pretrained=True)
        self.liner = nn.Linear(1000, num_class)
    def forward(self,x,drop_rate):
        x = self.net(x)
        x = F.relu(x)
        x = F.dropout(x,drop_rate)
        x = self.liner(x)
        return x


def train_model(model,dataloders,scriterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            i_count = 0
            total_count = int(dataset_sizes[phase]/batch_size)
            
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'train':
                    outputs = model(inputs,0.5)
                if phase == 'val':
                    outputs = model(inputs,0.0)
                    
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                i_count +=1
                print("[%4d:%4d] loss = %f"%(i_count,total_count,loss.data[0]))
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = Net(num_class=80)
model_ft = torch.nn.DataParallel(model_ft,device_ids=[0,1, 2,3,4,5])

if use_gpu:
    model_ft = model_ft.cuda()


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.7)


model_ft = train_model(model_ft,dataloder, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=20)

torch.save(model_ft.state_dict(),'best.mdl')
