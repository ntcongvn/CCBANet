from torch import optim

def GetLearningRateSchedule(config,optimizer):
  if config["train"]["learningschedule"]=="LambdaLR":
    var_nepoch=config["train"]["max_epochs"]
    var_power=config["train"]["schedule_lambdalr"]["power"]
    lr_lambda = lambda epoch: 1.0 - pow((epoch / var_nepoch), var_power)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
  elif config["train"]["learningschedule"]=="MultiStepLR":
    milestones=config["train"]["schedule_multisteplr"]["learing_epoch_steps"]
    gamma=config["train"]["schedule_multisteplr"]["learing_rate_schedule_factor"]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=milestones,gamma=gamma)
  else:
    raise Exception("Schedule setting is not valid")
  return lr_scheduler