import copy
import json                    #Read jon
import cv2
import random 
import numpy as np
import torchvision.transforms as transforms
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
import tifffile


def GetImage_Mask_Transform_SpatialLevel():
  image_mask_transform_spatiallevel=A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.5, rotate_limit=90,border_mode=0, value=0,p=0.5),
  ],additional_targets={ "image1": "image","mask1": "mask"},p=1)
  return image_mask_transform_spatiallevel

def GetImage_Mask_Transform_RandomCrop(image_size):
  image_mask_transform_randomCrop=A.Compose([
    A.OneOf([
      A.RandomCrop(height=image_size[0],width=image_size[1],p=1),
  ],p=1)],additional_targets={ "image1": "image","mask1": "mask"},p=1)
  return image_mask_transform_randomCrop

def GetImage_Transform_PixelLevel():
  image_transform_pixellevel = A.Compose([
      #A.RandomBrightnessContrast(p=0.5),
  ],p=1)
  return image_transform_pixellevel


def OpenTiffImageAndConvertToPilImage(path):
  im=tifffile.imread(path)
  im = Image.fromarray(im)
  return im
#Load dataset information from json file
def LoadCVCClinicDBDataset(config):
  data=[]
  image_dir=config["dataset_dir"]+"/CVC-ClinicDB/Original"
  for i in np.arange(1,613):                 #CVC-ClinicDB has 612 images with name form 1.tif to 612.tif
    #Get height, width
    print(image_dir+"/"+str(i)+".tif")
    img= tifffile.imread(image_dir+"/"+str(i)+".tif")
    h=img.shape[0]
    w=img.shape[1]
    d={"image":str(i)+".tif","height": h, "width": w}
    data.append(d)
  return data


class CVCClinicDBDataset(Dataset):
  
  def __init__(self,config,data,mode="train",normalization=True,augmentation=False):
    super().__init__()
    self.config=config
    self.data=data
    self.normalization=normalization
    self.augmentation=augmentation
    
    self.mode=mode
    if self.mode=="train":
      self.image_size=config["train"]["image_size"]
    elif self.mode=="val":
      self.image_size=config["val"]["image_size"]
    elif self.mode=="test":
      self.image_size=config["test"]["image_size"]
    else:
      raise Exception("Mode setting is not valid")

    self.imagenet_mean=config["imagenet_mean"]
    self.imagenet_std=config["imagenet_std"]
    
    self.image_paths = [] 
    self.label_paths = []
    #self.bboxs=[]  

    self.image_mask_transform_spatiallevel=GetImage_Mask_Transform_SpatialLevel()
    self.image_mask_transform_randomCrop=GetImage_Mask_Transform_RandomCrop(self.image_size)
    self.image_transform_pixellevel=GetImage_Transform_PixelLevel()

    # Define list of image transformations
    label_transformation = [transforms.ToTensor()]
    image_transformation = [transforms.ToTensor()]
    if self.normalization:
        image_transformation.append(transforms.Normalize(self.imagenet_mean, self.imagenet_std))
    self.label_transformation = transforms.Compose(label_transformation)
    self.image_transformation = transforms.Compose(image_transformation)
        
    # Get all image paths and label paths from data
    for index in np.arange(len(self.data)):
      d=self.data[index]
      if self.mode=="train" or self.mode=="val" or self.mode=="test":
        self.image_paths.append(config["dataset_dir"]+"/CVC-ClinicDB/Original/"+d["image"])
        self.label_paths.append(config["dataset_dir"]+"/CVC-ClinicDB/Ground Truth/"+d["image"])        
        #self.bboxs.append(d["bbox"])
      else:
        raise Exception("Mode setting is not valid")
            
  def __len__(self):
      return len(self.image_paths)

  def GetLength(self):
      return self.__len__()

  def GetDataItemByIndex(self,index):
      return self.data[index]

  def GetImagePathByIndex(self,index):
      return self.image_paths[index]

  def __getitem__(self, index):
    if self.mode=="train":   
      # Read image
      image_path = self.image_paths[index]
      image_data = OpenTiffImageAndConvertToPilImage(image_path)   #Image.open(image_path).convert("RGB")
      
      # Read label
      label_path = self.label_paths[index]
      mask_data = OpenTiffImageAndConvertToPilImage(label_path) #Image.open(label_path).convert("RGB") 
      
      image_data=np.array(image_data)
      mask_data=np.array(mask_data)
      if self.augmentation is True:
        transformed=self.image_mask_transform_spatiallevel(image=image_data,mask=mask_data)
        image_data=transformed["image"]
        mask_data=transformed["mask"]
        image_data=self.image_transform_pixellevel(image=image_data)["image"]

      if random.uniform(0, 1)>0.1 and self.augmentation is True:
        image_data=A.Resize(height=288, width=384, interpolation=cv2.INTER_LINEAR,p=1)(image=image_data)["image"]
        mask_data=A.Resize(height=288, width=384, interpolation=cv2.INTER_NEAREST,p=1)(image=mask_data)["image"]
        transformed=self.image_mask_transform_randomCrop(image=image_data,mask=mask_data)
        image_data=transformed["image"]
        mask_data=transformed["mask"]
      else:
        image_data=A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_LINEAR,p=1)(image=image_data)["image"]
        mask_data=A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_NEAREST,p=1)(image=mask_data)["image"]

      image_data=Image.fromarray(image_data)
      mask_data=Image.fromarray(mask_data)
        
      image_data = self.image_transformation(image_data)
      mask_data = self.label_transformation(mask_data)[0,:,:][None,:,:]  
      return image_data,mask_data,index

    elif self.mode=="val" or self.mode=="test":   
      # Read image
      image_path = self.image_paths[index]
      image_data = OpenTiffImageAndConvertToPilImage(image_path) #Image.open(image_path).convert("RGB")
      # Read label
      label_path = self.label_paths[index]
      mask_data = OpenTiffImageAndConvertToPilImage(label_path) #Image.open(label_path).convert("RGB") 
      
      image_data=np.array(image_data)
      mask_data=np.array(mask_data)
      image_data=A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_LINEAR,p=1)(image=image_data)["image"]
      mask_data=A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=cv2.INTER_NEAREST,p=1)(image=mask_data)["image"]
      image_data=Image.fromarray(image_data)
      mask_data=Image.fromarray(mask_data)

      image_data = self.image_transformation(image_data)
      mask_data = self.label_transformation(mask_data)[0,:,:][None,:,:]
      return image_data,mask_data,index

    else:
      raise Exception("Mode setting is not valid")
