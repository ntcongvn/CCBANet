import numpy as np                                                #linear algebra
import copy

import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torchvision.transforms as transforms
import tifffile
import os

#Test Dataloader and Visualize data
def MergeImageForVisualization(image,mask=None,bbox_data=None):
  alpha=0.5
  color =np.array([1,1,1]).astype(np.float)     #Mask color
  b_color =np.array([0,0,1]).astype(np.float)   #Bbox color
  result=copy.deepcopy(np.array(image)).transpose(1,2,0).astype(np.float)
  if mask is not None:
    mask=copy.deepcopy(np.array(mask.permute(1,2,0))).astype(np.float)
    overlay = mask * color
    result=cv2.addWeighted(result, alpha, overlay,1-alpha,0)
  
  if bbox_data is not None:
    bbox=bbox_data["bbox"]
    h,w=image.size()[1:]
    for b in bbox:
      xmin=int(b["xmin"])
      ymin=int(b["ymin"])
      xmax=int(b["xmax"])
      ymax=int(b["ymax"])

      x_scale = float(w) / float(bbox_data["width"])
      y_scale = float(h) / float(bbox_data["height"])
      xmin = int(np.round(xmin * x_scale))
      ymin = int(np.round(ymin * y_scale))
      xmax = int(np.round(xmax * x_scale))
      ymax = int(np.round(ymax * y_scale))
      result = cv2.rectangle(result, (xmin, ymin), (xmax, ymax), b_color,1)
  return result


def Visualize_dataset(dataloader):
  # Let's examine some of images in train set
  for data_batch,mask_batch,index_batch in dataloader:
    images=data_batch
    masks=mask_batch
    nums_img=images.shape[0]
    cols=4
    rows=nums_img//cols
    fig=plt.figure(figsize=(cols*7,rows*7))
    fig.suptitle("Some Augmented Images In Train Set ", fontsize=24)
    for i in np.arange(rows):
      for j in np.arange(cols):
        index=(i*cols)+j
        img = images[index]
        mask=masks[index] 
        #bbox_data=tmp_val_dataset.GetDataItemByIndex(index_batch[index])
        img=MergeImageForVisualization(img,mask=mask) #,bbox_data=bbox_data)
        plt.subplot(rows,cols,index+1)
        plt.title(tmp_train_dataset.GetDataItemByIndex(index_batch[index])["image"])
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    break




def VisualizeImageWithMask(dir_mask,dir_visualize,test_dataset,dataset_name):
  print("Run mask-visualizing ....")
  color_mask=(0,255,0)
  alpha=0.3
  num_imgs=test_dataset.GetLength()
  image_paths=[]
  mask_paths=[]
  image_names=[]
  for i in np.arange(num_imgs):
    image_paths.append(test_dataset.GetImagePathByIndex(i))
    if dataset_name=="CVC-ClinicDB" or dataset_name=="CVC-EndoSceneStill":
      mask_paths.append(dir_mask+"/"+os.path.splitext(test_dataset.GetDataItemByIndex(i)["image"])[0]+".jpg")
    else:
      mask_paths.append(dir_mask+"/"+test_dataset.GetDataItemByIndex(i)["image"])
    image_names.append(test_dataset.GetDataItemByIndex(i)["image"])

  for index in np.arange(num_imgs): 
    if dataset_name=="CVC-ClinicDB":
      image_org = tifffile.imread(image_paths[index])[:, :, ::-1].copy() 
    else:
      image_org = cv2.imread(image_paths[index])
    
    mask_org = cv2.imread(mask_paths[index])
    overlay = image_org.copy()
    output = image_org.copy()
    overlay[mask_org[:,:,0]>=128] = color_mask
    output=cv2.addWeighted(overlay, alpha, output, 1 - alpha,0)
    print("Generate {}/{}:{}....".format(index+1,num_imgs,os.path.splitext(image_names[index])[0]+".jpg"))
    
    dir_path='/'.join((dir_visualize+'/'+image_names[index]).split('/')[0:-1])
    if os.path.exists(dir_path) == False:
      os.makedirs(dir_path)

    if dataset_name=="CVC-ClinicDB" or dataset_name=="CVC-EndoSceneStill":
      cv2.imwrite(dir_visualize+'/'+os.path.splitext(image_names[index])[0]+".jpg", output)
    else:
      cv2.imwrite(dir_visualize+'/'+image_names[index], output)
