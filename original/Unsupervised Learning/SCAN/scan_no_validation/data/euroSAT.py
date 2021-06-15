#creating a custom dataloader 
"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.transforms import Compose,ToTensor,Normalize
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2
 
class EuroSat(Dataset):
 
    def __init__(self, df, transform=None):
 
        super(EuroSat, self).__init__()
 
    
        
        self.transform = transform
        
        
        self.classes = ['Forest','SeaLake','PermanentCrop','Industrial','River','AnnualCrop','HerbaceousVegetation','Residential','Highway','Pasture']
 
        
        self.df = df
 
        
 
  
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
 
        img = self.get_image(index)
        img_size = img.size
        
        class_name = self.df['label'][index]
        target = self.classes.index(class_name)
 
 
        if self.transform is not None:
            img = self.transform(img)
        
        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        
        return out
 
    def get_image(self, index):
        img_path = self.df['path'][index]   
        img = Image.open(img_path)
        img = img.convert('RGB')
        return img
 
    #functions used for plotting images     
    def show_img(self,index,aug = False):
        img =  self.get_image(index)
 
        if aug:
          img = self.transform(img).permute(1,2,0)
        
        plt.imshow(img)
    
    # will create a batch x batch images 
    def plot_batch(self,batch = 8,aug = False,indices = None,size = (20,20)):
 
        
        n_images = len(self.df)
 
        if indices == None:
           indices = np.random.choice(range(n_images) , size= batch **2) 
 
        #creating an image array 
        labels = self.df['label'][indices].reset_index(drop = True)
    
        images = [self.get_image(i) for i in indices]
        
        fig,axis = plt.subplots(batch,batch,figsize = size)
 
 
        for i,ax in enumerate(axis.flatten()):
 
            if aug:
                ax.imshow(self.transform(images[i]).permute(1,2,0))
            else: 
                ax.imshow(images[i])
 
            ax.set_title(labels[i])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
             
    
    def __len__(self):
        return len(self.df)