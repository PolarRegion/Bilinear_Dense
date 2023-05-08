import time

import torch.nn as nn
import torch
from models.bilinear_dense121 import BDense121_fc,BDense121_all,BDense121_all_Attention,BDense121_all_CBAM
from models.bilinear_dense201 import BDense201_fc,BDense201_all,BDense201_all_Attention,BDense201_all_CBAM
from torchsummary import summary
from torchvision.models import densenet201,resnet50

start = time.time()
net = resnet50
inchannel = net.fc.in_features
summary(net, (3, 224, 224))
end = time.time()
print(end-start)