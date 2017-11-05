import os
from torch.utils.data import Dataset
from PIL import Image
import json

class Exampler(Dataset):
    def __init__(self,label_path,image_path,transforms=None):
        self.data_dict = {}
        with open(label_path, 'r') as f:
            label_list = json.load(f)
        for image in label_list:
            self.data_dict[image['image_id']] = int(image['label_id'])
            
        self.Length = len(self.data_dict)
        self.img_name = list(self.data_dict.keys())
        self.image_path = image_path
        self.transform = transforms
        
    def __len__(self):
        return self.Length
    
    def __getitem__(self, index):
        image_name = os.path.join(self.image_path,self.img_name[index])
        image = Image.open(image_name)
        label = self.data_dict[self.img_name[index]]
        
        
        if self.transform:
            image = self.transform(image)
        
        return (image,label)
    
