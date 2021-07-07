from sklearn.model_selection import train_test_split              # Used for split train & val dataset
from torch.utils.data import DataLoader
from .kvasirseg import LoadKvasirSegDataset,KvasirSegDataset
from .cvcclinicdb import LoadCVCClinicDBDataset,CVCClinicDBDataset
from .cvcendoscene import LoadCVCEndoSceneDataset,CVCEndoSceneDataset

#Load dataset information
def LoadDataset(config):
  if config["dataset"]["name"]=="Kvasir-SEG":
    data=LoadKvasirSegDataset(config)
    #Split data
    random_state=config["dataset"]["random_state"]
    split_ratio_train_testval=config["dataset"]["split_ratio_train_testval"]
    split_ratio_test_val=config["dataset"]["split_ratio_test_val"]
    train_data, val_test_data = train_test_split(data, test_size=split_ratio_train_testval, random_state=random_state)
    val_data, test_data = train_test_split(val_test_data, test_size=split_ratio_test_val, random_state=random_state)
    return train_data, val_data, test_data
  elif config["dataset"]["name"]=="CVC-ClinicDB":
    data=LoadCVCClinicDBDataset(config)
    #Split data
    random_state=config["dataset"]["random_state"]
    split_ratio_train_testval=config["dataset"]["split_ratio_train_testval"]
    split_ratio_test_val=config["dataset"]["split_ratio_test_val"]
    train_data, val_test_data = train_test_split(data, test_size=split_ratio_train_testval, random_state=random_state)
    val_data, test_data = train_test_split(val_test_data, test_size=split_ratio_test_val, random_state=random_state)
    return train_data, val_data, test_data
  elif config["dataset"]["name"]=="CVC-EndoSceneStill":
    train_data,val_data,test_data=LoadCVCEndoSceneDataset(config)
    return train_data, val_data, test_data
  else:
    raise Exception("Dataset name is unvalid.")


def BuildDatasetAndDataloader(config,train_data,val_data,test_data,debug=False):
  if config["dataset"]["name"]=="Kvasir-SEG":
    train_dataset = KvasirSegDataset(config=config,data=train_data,mode="train",normalization=not debug,augmentation=config["train"]["image_data_augmentation"])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = KvasirSegDataset(config=config,data=val_data,mode="val",normalization=not debug,augmentation=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config["val"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    test_dataset = KvasirSegDataset(config=config,data=test_data,mode="test",normalization=not debug,augmentation=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["test"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    return train_dataset,train_dataloader,val_dataset,val_dataloader,test_dataset,test_dataloader

  elif config["dataset"]["name"]=="CVC-ClinicDB":
    train_dataset = CVCClinicDBDataset(config=config,data=train_data,mode="train",normalization=not debug,augmentation=config["train"]["image_data_augmentation"])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = CVCClinicDBDataset(config=config,data=val_data,mode="test",normalization=not debug,augmentation=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config["val"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    test_dataset = CVCClinicDBDataset(config=config,data=test_data,mode="val",normalization=not debug,augmentation=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["test"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    return train_dataset,train_dataloader,val_dataset,val_dataloader,test_dataset,test_dataloader
  
  elif config["dataset"]["name"]=="CVC-EndoSceneStill":
    train_dataset = CVCEndoSceneDataset(config=config,data=train_data,mode="train",normalization=not debug,augmentation=config["train"]["image_data_augmentation"])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = CVCEndoSceneDataset(config=config,data=val_data,mode="val",normalization=not debug,augmentation=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config["val"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    test_dataset = CVCEndoSceneDataset(config=config,data=test_data,mode="test",normalization=not debug,augmentation=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["test"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    return train_dataset,train_dataloader,val_dataset,val_dataloader,test_dataset,test_dataloader
  else:
    raise Exception("Dataset name is unvalid.")