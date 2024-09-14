import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class QuadrantSqueezeExcitation(nn.Module):
  def __init__(self, in_channels, reduction_ratio=0.25):
    super(QuadrantSqueezeExcitation, self).__init__()
    self.avgpool = nn.AdaptiveAvgPool2d((2,2))
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels*reduction_ratio), kernel_size=(1,1), stride=(1,1))
    self.conv2 = nn.Conv2d(in_channels=int(in_channels*reduction_ratio), out_channels=in_channels, kernel_size=(1,1), stride=(1,1))
    self.activation = nn.Hardswish()
    self.scale_activation = nn.Hardsigmoid()
  def forward(self, x):
    residual = x.clone()
    b,c,h,w = x.size()
    x = self.avgpool(x)
    x = self.conv1(x)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.scale_activation(x)
    # top left
    residual[:,:,0:int(h/2),0:int(w/2)] = residual[:,:,0:int(h/2),0:int(w/2)].clone() * x[:,:,0:1,0:1]
    # top right
    residual[:,:,0:int(h/2),int(w/2):w] = residual[:,:,0:int(h/2),int(w/2):w].clone() * x[:,:,0:1,1:2]
    # bottom left
    residual[:,:,int(h/2):h,0:int(w/2)] = residual[:,:,int(h/2):h,0:int(w/2)].clone() * x[:,:,1:2,0:1]
    # bottom right
    residual[:,:,int(h/2):h,int(w/2):w] = residual[:,:,int(h/2):h,int(w/2):w].clone() * x[:,:,1:2,1:2]
    return residual
  
class MobileNetV3Small(nn.Module):
  def __init__(self, num_classes):
    super(MobileNetV3Small, self).__init__()
    self.model = mobilenet_v3_small(weights='IMAGENET1K_V1', progress=True)
    self.model.classifier[3].out_features = num_classes
  def forward(self, x):
    return self.model(x)
  
class MobileNetV3SmallQuadrant(nn.Module):
  def __init__(self, num_classes):
    super(MobileNetV3SmallQuadrant, self).__init__()
    self.model = mobilenet_v3_small(weights='IMAGENET1K_V1', progress=True)
    qse_1 = QuadrantSqueezeExcitation(16)
    qse_2 = QuadrantSqueezeExcitation(96)
    qse_3 = QuadrantSqueezeExcitation(240)
    qse_4 = QuadrantSqueezeExcitation(240)
    qse_5 = QuadrantSqueezeExcitation(120)
    qse_6 = QuadrantSqueezeExcitation(144)
    qse_7 = QuadrantSqueezeExcitation(288)
    qse_8 = QuadrantSqueezeExcitation(576)
    qse_9 = QuadrantSqueezeExcitation(576)
    self.model.features[1].block[1] = qse_1
    self.model.features[4].block[2] = qse_2
    self.model.features[5].block[2] = qse_3
    self.model.features[6].block[2] = qse_4
    self.model.features[7].block[2] = qse_5
    self.model.features[8].block[2] = qse_6
    self.model.features[9].block[2] = qse_7
    self.model.features[10].block[2] = qse_8
    self.model.features[11].block[2] = qse_9
    self.model.classifier[3].out_features = num_classes
  def forward(self, x):
    return self.model(x)