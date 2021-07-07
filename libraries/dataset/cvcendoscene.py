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
      A.HueSaturationValue(p=0.3),
  ],p=1)
  return image_transform_pixellevel


def OpenTiffImageAndConvertToPilImage(path):
  im=tifffile.imread(path)
  im = Image.fromarray(im).convert("L")
  return im

#Load dataset information from json file
def LoadCVCEndoSceneDataset(config):
  image_dir=config["dataset_dir"]+"/CVC-EndoSceneStill"
  train_data=[]
  val_data=[]
  test_data=[]

  #Training:
  #CVC-300: 1-76,98-148,221-273
  #CVC-612: 26-50,104-126,178-227,253-317,384-503,529-612
  train_data.extend(["CVC-300/bbdd/"+str(i)+".bmp" for i in np.arange(1,76+1)])
  train_data.extend(["CVC-300/bbdd/"+str(i)+".bmp" for i in np.arange(98,148+1)])
  train_data.extend(["CVC-300/bbdd/"+str(i)+".bmp" for i in np.arange(221,273+1)])
  
  train_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(26,50+1)])
  train_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(104,126+1)])
  train_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(178,227+1)])
  train_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(253,317+1)])
  train_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(384,503+1)])
  train_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(529,612+1)])

  #Validation:
  #CVC-300: 77-97, 209-220,274-300
  #CVC-612: 51-103,228-252,318-342,364-383
  val_data.extend(["CVC-300/bbdd/"+str(i)+".bmp" for i in np.arange(77,97+1)])
  val_data.extend(["CVC-300/bbdd/"+str(i)+".bmp" for i in np.arange(209,220+1)])
  val_data.extend(["CVC-300/bbdd/"+str(i)+".bmp" for i in np.arange(274,300+1)])

  val_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(51,103+1)])
  val_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(228,252+1)])
  val_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(318,342+1)])
  val_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(364,383+1)])

  #Testing:
  #CVC-300: 149-208
  #CVC-612: 1-25, 127-177,343-363,504-528
  test_data.extend(["CVC-300/bbdd/"+str(i)+".bmp" for i in np.arange(149,208+1)])

  test_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(1,25+1)])
  test_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(127,177+1)])
  test_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(343,363+1)])
  test_data.extend(["CVC-612/bbdd/"+str(i)+".bmp" for i in np.arange(504,528+1)])

  print(train_data[0])
  for i in np.arange(len(train_data)):
    print(image_dir+"/"+train_data[i])
    img=Image.open(image_dir+"/"+train_data[i])
    w, h = img.size
    d={"image":train_data[i],"height": h, "width": w}
    train_data[i]=d

  for i in np.arange(len(val_data)):
    print(image_dir+"/"+val_data[i])
    img=Image.open(image_dir+"/"+val_data[i])
    w, h = img.size
    d={"image":val_data[i],"height": h, "width": w}
    val_data[i]=d

  for i in np.arange(len(test_data)):
    print(image_dir+"/"+test_data[i])
    img=Image.open(image_dir+"/"+test_data[i])
    w, h = img.size
    d={"image":test_data[i],"height": h, "width": w}
    test_data[i]=d

  return train_data,val_data,test_data


class CVCEndoSceneDataset(Dataset):
    
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

        self.image_paths.append(config["dataset_dir"]+"/CVC-EndoSceneStill/"+d["image"])
        if d["image"].find("CVC-612")==-1: # it is CVC-300 dataset use .bmp
          self.label_paths.append(config["dataset_dir"]+"/CVC-EndoSceneStill/"+(d["image"].replace("bbdd", "gtpolyp")))
        else: #it is CVC-612 use .tif
          self.label_paths.append(config["dataset_dir"]+"/CVC-EndoSceneStill/"+(d["image"].replace("bbdd", "gtpolyp").replace(".bmp", ".tif")))
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
      image_data = Image.open(image_path).convert("RGB")
      
      # Read label
      label_path = self.label_paths[index]
      if label_path.find(".tif")>-1:
        mask_data = OpenTiffImageAndConvertToPilImage(label_path) #tif image file
      else:
        mask_data = Image.open(label_path).convert("L")
      
      image_data=np.array(image_data)
      mask_data=np.array(mask_data)
      transformed=self.image_mask_transform_spatiallevel(image=image_data,mask=mask_data)
      image_data=transformed["image"]
      mask_data=transformed["mask"]
      image_data=self.image_transform_pixellevel(image=image_data)["image"]

      if random.uniform(0,1)>0.5 and self.augmentation is True:
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
      image_data = Image.open(image_path).convert("RGB")
      # Read label
      label_path = self.label_paths[index]
      if label_path.find(".tif")>-1:
        mask_data = OpenTiffImageAndConvertToPilImage(label_path) #tif image file
      else:
        mask_data = Image.open(label_path).convert("RGB") 
      
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
