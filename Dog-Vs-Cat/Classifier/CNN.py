

import torch
import torch.nn as nn

class VanilaCNN(nn.Module):
    def __init__(self):
        super(VanilaCNN, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=16,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # Layer 5
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # Layer 6
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
             # Layer 7
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # Layer 8
            nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            # Layer 9
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            
        )
        self.net = nn.Linear(512, 512)
        self.net = nn.Linear(512, 2)
    def forward(self, x):
        return self.net(self.conv(x).squeeze())

