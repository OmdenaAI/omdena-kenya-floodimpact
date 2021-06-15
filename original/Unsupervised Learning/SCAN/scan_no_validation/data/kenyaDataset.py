from PIL import Image
from torch.utils.data import Dataset 
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2
import numpy as np

class KenyaDataset(Dataset):

    def __init__(self,img_path,transform):

      self.img_path = img_path 

      self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.img_path))

    def get_image(self, index):
        img_path = os.path.join(self.img_path,f'image_{index}.jpeg')  
        img = Image.open(img_path)
        img = img.convert('RGB')
        return img
 
    def __getitem__(self,index):

      img = self.get_image(index)

      if self.transform:
          img = self.transform(img)

      return {'image':img}

    #functions used for plotting images     
    def show_img(self,index,aug = False):
        img =  self.get_image(index)
 
        if aug:
          img = self.transform(img).permute(1,2,0)
        
        plt.imshow(img)
    
    # will create a batch x batch images 
    def plot_batch(self,batch = 8,indices = None,size = (20,20)):
 
        
        if not self.transform:
           raise ValueError("Transforms are none. ") 


        n_images = self.__len__()
 
        if indices == None:
           indices = np.random.choice(range(n_images) , size=  batch * (batch //2)) 
 
     
        images = [self.get_image(i) for i in indices]
        
        fig,axis = plt.subplots(batch,batch,figsize = size)
 
        aug = False
        idx = 0
        for  ax in axis.flatten():
 
            if  not aug:
              ax.imshow(images[idx])
              aug = True
            else:
              ax.imshow(self.transform(images[idx]).permute(1,2,0))
              idx += 1
              aug = False

                
 
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
             
    
