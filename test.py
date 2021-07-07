import sys
sys.path.append("./libraries/CCBANet")
import os 
import argparse
import numpy as np
import torch
import time
from config import CONFIG
from libraries.dataset import LoadDataset 
from libraries.dataset import BuildDatasetAndDataloader
from libraries.CCBANet.models.CCBANet import CCBANetModel
from libraries.CCBANet.utils.loss import DeepSupervisionLoss
from libraries.CCBANet.utils.metrics import evaluate,evaluate_single
from libraries.utils.optimizer import GetOptimizer
from libraries.utils.schedule import GetLearningRateSchedule
from libraries.utils.visualize import VisualizeImageWithMask

import torchvision.transforms as transforms
import tifffile
from PIL import Image
import cv2


def parse_arguments():
  parse = argparse.ArgumentParser(description='CCBANet Polyp Segmentation')

  parse.add_argument('--dataset', type=str, default='Kvasir-SEG')
  parse.add_argument('--batch_size', type=int, default=1)
  parse.add_argument('--use_gpu', type=bool, default=True)
  parse.add_argument('--load_ckpt', type=str, default=None)
  
  return parse.parse_args()

def update_config(CONFIG,opt):
  CONFIG["mode"]== "test"
  CONFIG["dataset"]["name"]=opt.dataset
  CONFIG["train"]["batch_size"]=opt.batch_size
  if opt.use_gpu==True:
    CONFIG["device"]="cuda"
  else:
    CONFIG["device"]="cpu"
  if opt.load_ckpt is not None:
    CONFIG["test"]["pretrain_model"]=opt.load_ckpt


def epoch_testing(model, test_dataloader, device,criteria_metrics):
  # Switch model to evaluation mode
  total_time=0
  total=len(test_dataloader)
  model.eval()

  out_pred = torch.FloatTensor().to(device)      # Tensor stores prediction values
  out_gt = torch.FloatTensor().to(device)        # Tensor stores groundtruth values
  out_index = torch.IntTensor()
  with torch.no_grad(): # Turn off gradient
    # For each batch
    for step, (images,masks,indexs) in enumerate(test_dataloader):
      out_index = torch.cat((out_index, indexs), 0)
      # Move images, labels to device (GPU)
      images = images.to(device)
      masks = masks.to(device)
      indexs = indexs.to(device)
      out_gt = torch.cat((out_gt,  masks), 0)

      # Feed forward the model
      test_time_begin=time.time()
      predicts = model(images)
      test_time_end=time.time()
      total_time+=(test_time_end-test_time_begin)
      
      out_pred = torch.cat((out_pred, predicts[0]), 0)
      print('Testing iter:{:d}/{:d}'.format(step+1,total))
  
  _recall, _specificity, _precision, _F1, _F2, _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean=criteria_metrics(out_pred, out_gt)
  score_metrics={
      "recall":_recall,
       "specificity":_specificity,
       "precision":_precision,
       "f1":_F1,
       "f2":_F2,
       "accuracy":_ACC_overall,
       "iou_poly":_IoU_poly,
       "iou_bg":_IoU_bg,
       "iou_mean":_IoU_mean
  }
  del images,masks
  if torch.cuda.is_available(): torch.cuda.empty_cache()
  return out_pred,out_index,score_metrics,total_time



if __name__ == '__main__':
    opt=parse_arguments()
    update_config(CONFIG,opt)
    print(CONFIG)

    train_data, val_data, test_data=LoadDataset(CONFIG)
    #Create dataset and dataloader
    _,_,_,_,test_dataset,test_dataloader=BuildDatasetAndDataloader(CONFIG,train_data,val_data,test_data)
    
    #Create model and get number of trainable parameters
    model = CCBANetModel(CONFIG).to(CONFIG["device"])
    print(model)
    #Number of trainable parameters
    print("Num of patameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    
    dir_mask=CONFIG["segm_dir"]
    dir_visualize=CONFIG["visualize_dir"]
    device=CONFIG["device"]
    path_model= CONFIG["test"]["pretrain_model"]

    model.load_state_dict(torch.load(path_model,map_location=torch.device(device))["model"])
    model.eval()

    #Metrics
    criteria_metrics=evaluate_single #evaluate
    
    print("Run testing....")
    out_pred,out_index,score_metrics,total_time=epoch_testing(model, test_dataloader, device, criteria_metrics)

    #printf info
    score_dicecoef=score_metrics["f1"]
    score_f2=score_metrics["f2"]
    score_precision=score_metrics["precision"]
    score_accuracy=score_metrics["accuracy"]
    score_recall=score_metrics["recall"]
    score_iou_poly=score_metrics["iou_poly"] 
    score_iou_bg=score_metrics["iou_bg"] 
    score_iou_mean=score_metrics["iou_mean"] 
    score_specificity=score_metrics["specificity"] 
    print('Dice:{:2.4f} | IoU-poly:{:2.4f} | IoU-bg:{:2.4f} | IoU-mean:{:2.4f} | AP:{:2.4f} | AR:{:2.4f} | F2:{:2.4f} | ACC:{:2.4f}'.format(score_dicecoef,score_iou_poly,score_iou_bg,score_iou_mean,score_precision,score_recall,score_f2,score_accuracy))
    num_imgs=out_index.size()[0]
    fps=num_imgs/total_time
    meantime=total_time/num_imgs
    print('Number of images:{:4d} | Total time:{:4.4f} | Mean time:{:4.4f} | Frame per second:{:2.4f}'.format(num_imgs,total_time,meantime,fps))
    
    if CONFIG["test"]["mask_generating"]==True:
      print("Run mask-generating ....")
      #Generate mask
      trans_pil=transforms.ToPILImage()
      for index in np.arange(num_imgs):
        img=out_pred[index].cpu()
        img=(img>0.5)*1.0
        img=torch.cat([img,img,img],0)
        metadata=test_dataset.GetDataItemByIndex(out_index[index])
        image_name=metadata["image"]
        w=int(metadata["width"])
        h=int(metadata["height"])
        trans_size= transforms.Resize((h,w),Image.NEAREST)
        img=trans_size(trans_pil(img))
        
        dir_path='/'.join((dir_mask+"/"+image_name).split('/')[0:-1])
        if os.path.exists(dir_path) == False:
          os.makedirs(dir_path)

        if CONFIG["dataset"]["name"]=="CVC-ClinicDB" or CONFIG["dataset"]["name"]=="CVC-EndoSceneStill":
          img.save(dir_mask+"/"+os.path.splitext(image_name)[0]+".jpg")
        else:
          img.save(dir_mask+"/"+image_name)
        print("Generate {}/{}:{} (w:{} x h:{})".format(index+1,num_imgs,os.path.splitext(image_name)[0]+".jpg",w,h))
      
      if CONFIG["test"]["mask_visualize"]==True:
        VisualizeImageWithMask(dir_mask,dir_visualize,test_dataset,CONFIG["dataset"]["name"])

    print('Done')

