from torch import optim

def GetOptimizer(config,model):
  #Optimizer
  # load optimizer and scheduler
  if config["train"]["optimizer"]=="SGD":
    lr=config["train"]["learing_rate"]
    momentum=config["train"]["optimizer_sgd"]["momentum"]
    weight_decay=config["train"]["optimizer_sgd"]["weight_decay"]
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  elif config["train"]["optimizer"]=="Adam":
    lr=config["train"]["learing_rate"]
    betas=config["train"]["optimizer_adam"]["betas"]
    eps=config["train"]["optimizer_adam"]["eps"]
    weight_decay=config["train"]["optimizer_adam"]["weight_decay"]
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
  else:
    raise Exception("Optimzer setting is not valid")
  return optimizer