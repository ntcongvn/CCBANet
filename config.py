CURRENT_DIR="."
CONFIG={
    "dataset_dir":CURRENT_DIR+"/datasets",
    "output_dir":CURRENT_DIR+"/output",
    "model_dir":CURRENT_DIR+"/output/model",
    "segm_dir":CURRENT_DIR+"/output/segm",
    "visualize_dir":CURRENT_DIR+"/output/visualize",
    "dataset":{
        "name":"Kvasir-SEG",                                      # "Kvasir-SEG", "CVC-ClinicDB", "CVC-EndoSceneStill"
        "random_state":2021,
        "split_ratio_train_testval":0.2,
        "split_ratio_test_val":0.5,
    },                                     
    "device":"cuda",                                             #"cuda", "cpu"
    "mode":"train",                                              #"train", "test", "val"
    "typeloss":"BceDiceLoss",                                    #"StructureLoss", "BceDiceLoss", "BceIoULoss"
    "imagenet_mean":[0.485, 0.456, 0.406],
    "imagenet_std":[0.229, 0.224, 0.225],
    "model":{
        "num_classes":1,
    },
    "resume":{
        "is_resume":False,
        "epoch_start":0,
        "pretrain_model":"",
    },
    "train":{
        "image_size":(256,256),                                 #h w 
        "image_data_augmentation":True,
        "batch_size":8,                                         # 14 (224x224) ; 8 (256x256) for "Kvasir-SEG", "CVC-ClinicDB"; "CVC-EndoSceneStill" with 16G GPU
        "optimizer":"Adam",                                      #"SGD", "Adam"
        "learningschedule":"LambdaLR",                          #"LambdaLR", "MultiStepLR"
        "optimizer_sgd":{
            "momentum":0.9,
            "weight_decay":1e-5,
        },
        "optimizer_adam":{
            "betas":(0.9, 0.999),
            "eps":1e-8,
            "weight_decay":1e-5,
        },
        "schedule_lambdalr":{
            "power":0.9
        },
        "schedule_multisteplr":{
            "learing_rate_schedule_factor":0.1,
            "learing_epoch_steps":[80,150,180],
        },
        "learing_rate":0.0001,
        "max_epochs":200,
        "training_time_out":3600*24*3,
        "early_stop":200                         #50
    },
    "val":{
        "image_size":(320,320),                  #h w  (320x320) for "Kvasir-SEG", (288x384) for "CVC-ClinicDB"; (288x384) for "CVC-EndoSceneStill" with 16G GPU  
        "batch_size":1,
        "pretrain_model":"./output/model/model_epoch_143_score_0.84663.pth"
    },
    "test":{
        "image_size":(320,320),                  #h w  (320x320) for "Kvasir-SEG", (288x384) for "CVC-ClinicDB"; (288x384) for "CVC-EndoSceneStill" with 16G GPU 
        "batch_size":1, 
        "pretrain_model":"",
        "mask_generating":True,
        "mask_visualize":True
    },
}