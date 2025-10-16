import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=5) # I : 1 image 28x28 / O : 8 images 24x24
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5) # I : 8 images 24x24
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(in_features=16*4*4,out_features=128)
        self.fc2 = nn.Linear(in_features=128,out_features=64)
        self.fc3 = nn.Linear(in_features=64,out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # I : 1 image 28x28 / O : 8 canaux 24x24     
        x = self.pool(x)           # I : 8 canaux 24x24 / O : 8 canaux 12x12    
        x = F.relu(self.conv2(x))  # I : 8 canaux 12x12 / O : 16 canaux 8x8
        x = self.pool(x)           # I : 16 canaux 8x8 / O : 16 canaux 4x4
        x = torch.flatten(x,1)     # I : 16 canaux 4x4 / O : 1 canal 16x4x4 = 256
        x = F.relu(self.fc1(x))    # 256 --> 128
        x = F.relu(self.fc2(x))    # 128 --> 64
        x = self.fc3(x)            # 64 --> 10
        return x
    
    def get_features(self,x): # to get the embeddings computed after the last conv layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) #same as x.view(-1,16*4*4)
        return x