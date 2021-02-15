from torch.utils.data import Dataset
import pandas as pd
import os
import torchvision.transforms as transforms 
from PIL import Image
import torch
import cv2
import numpy as np

transform = transforms.Compose([    
                                transforms.Resize((56,56), interpolation = 2),
                                transforms.ToTensor(),
                                ])

class Breast_cancer_dataset(Dataset):
    def __init__(self, df_data,transform=None):
        super().__init__()
        self.df = df_data.values # Dataframe.values
        
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path,label = self.df[index]
        #print(img_path,label)
        image = cv2.imread(img_path) # Numpy array 
        #print(image.shape) # (50,50,3)
        if self.transform is not None:
            image = self.transform(image)
        if index % 1000 == 0 :
            print(index)
        return image, label
    
# class Breast_cancer_dataset(Dataset):
#     def __init__(self, root_dir, annotation_file, transform=None):
#         self.root_dir = root_dir
#         self.annotations = pd.read_csv(annotation_file)
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         img_id = self.annotations.iloc[index, 0]
#         img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
#         y_label = torch.tensor(float(self.annotations.iloc[index, 1]))

#         if self.transform is not None:
#             img = self.transform(img)

#         return (img, y_label)
