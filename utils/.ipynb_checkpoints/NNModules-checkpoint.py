import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, efficientnet_v2_s, efficientnet_b0, efficientnet_b3


class QuadrantSqueezeExcitation(nn.Module):
  def __init__(self, in_channels, reduction_dim):
    super(QuadrantSqueezeExcitation, self).__init__()
    self.avgpool = nn.AdaptiveAvgPool2d((2,2))
    self.maxpool = nn.AdaptiveMaxPool2d((2,2))
    self.fc = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=reduction_dim, kernel_size=(1,1), stride=(1,1)),
        nn.Hardswish(),
        nn.Conv2d(in_channels=reduction_dim, out_channels=in_channels, kernel_size=(1,1), stride=(1,1))
    )
    self.scale_activation = nn.Hardsigmoid()
      
  def forward(self, x):
    residual = x.clone()
    b,c,h,w = x.size()
    avg_out = self.fc(self.avgpool(x))
    max_out = self.fc(self.maxpool(x))
    x = avg_out + max_out  
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
    qse_1 = QuadrantSqueezeExcitation(16,8)
    qse_2 = QuadrantSqueezeExcitation(96,24)
    qse_3 = QuadrantSqueezeExcitation(240,64)
    qse_4 = QuadrantSqueezeExcitation(240,64)
    qse_5 = QuadrantSqueezeExcitation(120,32)
    qse_6 = QuadrantSqueezeExcitation(144,40)
    qse_7 = QuadrantSqueezeExcitation(288,72)
    qse_8 = QuadrantSqueezeExcitation(576,144)
    qse_9 = QuadrantSqueezeExcitation(576,144)
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
  

class EfficientNetB0(nn.Module):
  def __init__(self, num_classes):
    super(EfficientNetB0, self).__init__()
    self.model = efficientnet_b0(weights='IMAGENET1K_V1', progress=True)
    self.model.classifier[1].out_features = num_classes
  def forward(self, x):
    return self.model(x)
  
  
class EfficientNetB0Quadrant(nn.Module):
  def __init__(self, num_classes, all=False):
    super(EfficientNetB0Quadrant, self).__init__()
    self.model = efficientnet_b0(weights='IMAGENET1K_V1', progress=True)
    self.model.classifier[1].out_features = num_classes
    # qse in stage 2
    self.model.features[1][0].block[1] = QuadrantSqueezeExcitation(32,8)
    # qse in stage 3
    self.model.features[2][0].block[2] = QuadrantSqueezeExcitation(96,4)
    self.model.features[2][1].block[2] = QuadrantSqueezeExcitation(144,6)
    # qse in stage 4
    self.model.features[3][0].block[2] = QuadrantSqueezeExcitation(144,6)
    self.model.features[3][1].block[2] = QuadrantSqueezeExcitation(240,10)
    if all:
      # qse in stage 5
      self.model.features[4][0].block[2] = QuadrantSqueezeExcitation(240,10)
      for i in range(1,3):
        self.model.features[4][i].block[2] = QuadrantSqueezeExcitation(480,20)
      # qse in stage 6
      self.model.features[5][0].block[2] = QuadrantSqueezeExcitation(480,20)
      for i in range(1,3):
        self.model.features[5][i].block[2] = QuadrantSqueezeExcitation(672,28)
      # qse in stage 7
      self.model.features[6][0].block[2] = QuadrantSqueezeExcitation(672,28)
      for i in range(1,4):
        self.model.features[6][i].block[2] = QuadrantSqueezeExcitation(1152,48)
      # qse in stage 8
      self.model.features[7][0].block[2] = QuadrantSqueezeExcitation(1152,48)
  def forward(self, x):
    return self.model(x)
  
  
class EfficientNetB3(nn.Module):
  def __init__(self, num_classes):
    super(EfficientNetB3, self).__init__()
    self.model = efficientnet_b3(weights="IMAGENET1K_V1", progress=True)
    self.model.classifier[1].out_features = num_classes
  def forward(self, x):
    return self.model(x)
  
  
class EfficientNetB3Quadrant(nn.Module):
  def __init__(self, num_classes, qse_stages):
    super(EfficientNetB3Quadrant, self).__init__()
    self.model = efficientnet_b3(weights="IMAGENET1K_V1", progress=True)
    self.model.classifier[1].out_features = num_classes
    # qse in stage 2
    if 2 in qse_stages:
        self.model.features[1][0].block[1] = QuadrantSqueezeExcitation(40,10)
        self.model.features[1][1].block[1] = QuadrantSqueezeExcitation(24,6)
    # qse in stage 3
    if 3 in qse_stages:
        self.model.features[2][0].block[2] = QuadrantSqueezeExcitation(144,6)
        for i in range(1,3):
          self.model.features[2][1].block[2] = QuadrantSqueezeExcitation(192,8)
    # qse in stage 4
    if 4 in qse_stages:
      self.model.features[3][0].block[2] = QuadrantSqueezeExcitation(192,8)
      for i in range(1,3):  
        self.model.features[3][i].block[2] = QuadrantSqueezeExcitation(288,12)
    # qse in stage 5
    if 5 in qse_stages:
      self.model.features[4][0].block[2] = QuadrantSqueezeExcitation(288,12)
      for i in range(1,5):
        self.model.features[4][i].block[2] = QuadrantSqueezeExcitation(576,24)
    # qse in stage 6
    if 6 in qse_stages:
      self.model.features[5][0].block[2] = QuadrantSqueezeExcitation(576,24)
      for i in range(1,5):
        self.model.features[5][i].block[2] = QuadrantSqueezeExcitation(816,34)
    # qse in stage 7
    if 7 in qse_stages:
      self.model.features[6][0].block[2] = QuadrantSqueezeExcitation(816,34)
      for i in range(1,6):
        self.model.features[6][i].block[2] = QuadrantSqueezeExcitation(1392,58)
    # qse in stage 8
    if 8 in qse_stages:
      self.model.features[7][0].block[2] = QuadrantSqueezeExcitation(1392,58)
      self.model.features[7][1].block[2] = QuadrantSqueezeExcitation(2304,96)
  def forward(self, x):
    return self.model(x)
    

class EfficientNetV2Small(nn.Module):
  def __init__(self, num_classes):
    super(EfficientNetV2Small, self).__init__()
    self.model = efficientnet_v2_s(weights='IMAGENET1K_V1', progress=True)
    self.model.classifier[1].out_features = num_classes
  def forward(self, x):
    return self.model(x)
  
  
class EfficientNetV2SmallQuadrant(nn.Module):
  def __init__(self, num_classes):
    super(EfficientNetV2SmallQuadrant, self).__init__()
    self.model = efficientnet_v2_s(weights="IMAGENET1K_V1", progress=True)
    self.model.classifier[1].out_features = num_classes
    # qse in stage 4
    self.model.feautres[4][0].block[2] = QuadrantSqueezeExcitation(256,16)
    for i in range(1,6):
        self.model.feautres[4][i].block[2] = QuadrantSqueezeExcitation(512,32)
    # qse in stage 5
    self.model.features[5][0].block[2] = QuadrantSqueezeExcitation(758,32)
    for i in range(1,9):
      self.model.features[5][i].block[2] = QuadrantSqueezeExcitation(960,40)
    #qse in stage 6
    self.model.features[6][0].block[2] = QuadrantSqueezeExcitation(940,40)
    for i in range(1,15):
      self.model.features[6][i].block[2] = QuadrantSqueezeExcitation(1536,64)
  def forward(self, x):
    return self.model(x)
