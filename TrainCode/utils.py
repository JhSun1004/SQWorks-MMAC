import torchvision.models as models
import torch.nn as nn
import sys
from torch import optim

def get_model(name,num_class=5):
    if name == 'resnet50':
        net = models.resnet50(weights="IMAGENET1K_V2")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class) 
    elif name == 'resnet18':
        net = models.resnet18(weights="IMAGENET1K_V1")
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_class) 
    elif  name == 'efficientnetb1':
        net = models.efficientnet_b1(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)
    elif name == 'efficientnetb0':
        net = models.efficientnet_b0(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)
    elif name == 'efficientnetb2':
        net = models.efficientnet_b2(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class)
    elif name == 'efficientnetb3':
        net = models.efficientnet_b3(weights = "IMAGENET1K_V1")
        in_channel = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_channel, num_class) 
    else:
        print('no model found')
        sys.exit(0)
    return net

def get_lr_scheduler(name, optimizer):
    if name == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    elif name == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-9)
    else:
        sys.exit('lr_scheduler not found')
    return lr_scheduler
