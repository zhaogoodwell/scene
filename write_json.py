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
import json
from PIL import Image
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()



batch_size = 32
data_transforms = {
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

val_path_base   = "/home/ime/work/ai/dataset/ai_challenger_scene_validation_20170908/"
val_image_path  = os.path.join(val_path_base,"scene_validation_images_20170908")

test_path_base  = '/home/ime/work/ai/dataset/ai_challenger_scene_test_a_20170922'
test_image_path = os.path.join(test_path_base,"scene_test_a_images_20170922")

class Net(nn.Module):
    def __init__(self,num_class):
        super(Net,self).__init__()
        self.net = models.resnet152(pretrained=False)
        self.liner = nn.Linear(1000, num_class)
    def forward(self,x,drop_rate):
        x = self.net(x)
        x = F.relu(x)
        x = F.dropout(x,drop_rate)
        x = self.liner(x)
        return x

def predict_model(model,test_dir,image_transform,output):
    model.train(False)
    test_images = os.listdir(test_dir)
    result = []
    
    for test_image in test_images:
        temp_dict = {}
        
        img  = Image.open(os.path.join(test_dir, test_image))
        img  =  image_transform(img)
        img  = img.unsqueeze(0)
                
        img_test = Variable(img.cuda())
        
        predict = model(img_test,0.0)
        _,predictions = torch.topk(predict,3)
        temp_dict['image_id'] = test_image
        out = list(predictions.data[0])
        temp_dict['label_id'] = out
        result.append(temp_dict)
        print('image %s is %d,%d,%d' % (test_image, out[0],out[1],out[2]))
        
    with open(output+'submit.json', 'w') as f:
        json.dump(result, f)
        print('write result json, num is %d' % len(result))
    print("done")



model_ft = Net(80)

model_ft = torch.nn.DataParallel(model_ft,device_ids=[0])
model_ft = model_ft.cuda()

model_ft.load_state_dict(torch.load("best.mdl"))


predict_model(model_ft, val_image_path, data_transforms['val'],'val')
predict_model(model_ft, test_image_path, data_transforms['val'],'test')



