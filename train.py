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



def parse_arguments():
  parse = argparse.ArgumentParser(description='CCBANet Polyp Segmentation')

  parse.add_argument('--dataset', type=str, default='Kvasir-SEG')
  parse.add_argument('--batch_size', type=int, default=8)
  parse.add_argument('--use_gpu', type=bool, default=True)
  parse.add_argument('--load_ckpt', type=str, default=None)
  parse.add_argument('--epoch_start', type=int, default=0)
  

  return parse.parse_args()

def update_config(CONFIG,opt):
  CONFIG["mode"]== "train"
  CONFIG["dataset"]["name"]=opt.dataset
  CONFIG["train"]["batch_size"]=opt.batch_size
  if opt.use_gpu==True:
    CONFIG["device"]="cuda"
  else:
    CONFIG["device"]="cpu"
  if opt.load_ckpt is not None:
    CONFIG["resume"]["is_resume"]=True
    CONFIG["resume"]["epoch_start"]=opt.epoch_start
    CONFIG["resume"]["pretrain_model"]=opt.load_ckpt

#Define training excution for each epoch 
def epoch_training(model, train_dataloader, device, criteria_loss,typeloss, optimizer):
  # Switch model to training mode
  model.train()
  training_loss = 0 # Storing sum of training losses

  # For each batch
  for step, (images,masks,_) in enumerate(train_dataloader):
    #print(step*16,"/",len(train_dataloader)*16)
    # Move X, Y  to device (GPU)
    images = images.to(device)
    masks = masks.to(device)

    # Clear previous gradient
    optimizer.zero_grad()

    # Feed forward the model
    predicts = model(images)
    loss=criteria_loss(predicts, masks,typeloss)
    loss.backward()

    # Update parameters
    optimizer.step()

    # Update training loss after each batch
    training_loss += loss.item()

  del images,masks,loss
  if torch.cuda.is_available(): 
    torch.cuda.empty_cache()

  # return training loss
  return training_loss/len(train_dataloader)

#Define training evaluation for each epoch
def epoch_evaluating(model, val_dataloader, device, criteria_loss,criteria_metrics):
  # Switch model to evaluation mode
  model.eval()

  val_loss = 0                                   # Total loss of model on validation set
  out_pred = torch.FloatTensor().to(device)      # Tensor stores prediction values
  out_gt = torch.FloatTensor().to(device)        # Tensor stores groundtruth values

  with torch.no_grad(): # Turn off gradient
    # For each batch
    for step, (images, masks,_) in enumerate(val_dataloader):
      # Move images, labels to device (GPU)
      images = images.to(device)
      masks = masks.to(device)

      # Update groundtruth values
      out_gt = torch.cat((out_gt,  masks), 0)

      # Feed forward the model
      predicts= model(images)
      loss = criteria_loss(predicts, masks)

      # Update prediction values
      out_pred = torch.cat((out_pred, predicts[0]), 0)

      # Update validation loss after each batch
      val_loss += loss.item()    
 
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
  # Clear memory
  del images, masks, loss, out_pred, out_gt
  if torch.cuda.is_available(): torch.cuda.empty_cache()
  # return validation loss, and metric score
  return val_loss/len(val_dataloader), score_metrics

#Fully training

def full_training(config,model,train_dataloader,val_dataloader,optimizer,criteria_loss,criteria_metrics,lr_scheduler):
  device=config["device"]
  PATH_SAVE_MODEL=config["model_dir"]
  MAX_EPOCHS=config["train"]["max_epochs"]
  EARLY_STOP=config["train"]["early_stop"]
  TRAINING_TIME_OUT = config["train"]["training_time_out"]
  TYPELOSS=config["typeloss"]

  epoch_start=0
  if CONFIG["resume"]["is_resume"]== True:
    epoch_start=CONFIG["resume"]["epoch_start"]

  # Best Dice Coef value during training
  best_score = 0
  training_losses = []
  validation_losses = []
  validation_score = []
   
  nonimproved_epoch = 0
  start_time = time.time()

  # Training each epoch
  for epoch in np.arange(epoch_start,MAX_EPOCHS):
    
    # Training
    train_loss = epoch_training(model, train_dataloader, device, criteria_loss,TYPELOSS, optimizer)
    training_losses.append(train_loss)

    # Evaluating
    val_loss, score_metrics = epoch_evaluating(model, val_dataloader, device, criteria_loss,criteria_metrics)
    new_score=score_metrics["f1"]
    validation_losses.append(val_loss)
    validation_score.append(new_score)

    #printf info
    lr_current=lr_scheduler.get_last_lr()[0]
    score_dicecoef=score_metrics["f1"]
    score_f2=score_metrics["f2"]
    score_precision=score_metrics["precision"]
    score_accuracy=score_metrics["accuracy"]
    score_recall=score_metrics["recall"]
    score_iou_poly=score_metrics["iou_poly"] 
    score_iou_bg=score_metrics["iou_bg"] 
    score_iou_mean=score_metrics["iou_mean"] 
    score_specificity=score_metrics["specificity"] 

    print('Epoch:{:3d}/{:3d}|Train loss:{:2.4f} | Val loss:{:2.4f} | Lr:{:2.7f} | Dice:{:2.4f} | IoU-poly:{:2.4f} | IoU-bg:{:2.4f} | IoU-mean:{:2.4f} | AP:{:2.4f} | AR:{:2.4f} | F2:{:2.4f} | ACC:{:2.4f}'.format(epoch,MAX_EPOCHS,train_loss,val_loss,lr_current,score_dicecoef,score_iou_poly,score_iou_bg,score_iou_mean,score_precision,score_recall,score_f2,score_accuracy))
 
    # Update learning rate
    #lr_scheduler.step(new_score)
    lr_scheduler.step() #lr_scheduler.step(epoch)

    # Save model
    if best_score < new_score:
        print(f"Improve Dice Coef Core from {best_score} to {new_score}")
        best_score = new_score
        nonimproved_epoch = 0

        #Save model for each epoch
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "best_score": best_score, 
                    "epoch": epoch}, (PATH_SAVE_MODEL+'/model_epoch_{:d}.pth').format(epoch))

    else: 
        nonimproved_epoch += 1

    
    if nonimproved_epoch > EARLY_STOP:
        print("Early stopping")
        break
    if time.time() - start_time > TRAINING_TIME_OUT:
        print("Out of time")
        break


if __name__ == '__main__':
    opt=parse_arguments()
    update_config(CONFIG,opt)
    print(CONFIG)

    train_data, val_data, test_data=LoadDataset(CONFIG)

    #Create dataset and dataloader
    train_dataset,train_dataloader,val_dataset,val_dataloader,_,_=BuildDatasetAndDataloader(CONFIG,train_data,val_data,test_data)
    
    #Create model and get number of trainable parameters
    model = CCBANetModel(CONFIG).to(CONFIG["device"])
    print(model)
    #Number of trainable parameters
    print("Num of patameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
    #Binary Cross Entropy Dice Coefficient
    criteria_loss = DeepSupervisionLoss

    #Metrics
    criteria_metrics=evaluate_single #evaluate

    #Optimizer 
    optimizer=GetOptimizer(CONFIG,model)

    #Learning rete schedule
    lr_scheduler=GetLearningRateSchedule(CONFIG,optimizer)

    if CONFIG["resume"]["is_resume"]== True:
      device=CONFIG["device"]
      path_model= CONFIG["resume"]["pretrain_model"]
      print("Load:",path_model)
      checkpoint=torch.load(path_model,map_location=torch.device(device))
      model.load_state_dict(checkpoint["model"])
      optimizer.load_state_dict(checkpoint["optimizer"])
      lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
      del checkpoint

    full_training(CONFIG,model,train_dataloader,val_dataloader,optimizer,criteria_loss,criteria_metrics,lr_scheduler)
    print('Done')

