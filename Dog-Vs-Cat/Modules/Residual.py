import torch
import torch.nn as nn

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, activation = nn.ReLU):
        super(ResidualConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.relu1 = nn.ReLU(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=2)
        self.relu2 = nn.ReLU(out_channels)
        
        if in_channels != out_channels:
            self.projected = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            # self.projected = nn.Linear(in_channels*2*64*64, out_channels)

    def forward(self, x):
        identity = x
        print(identity.shape)

        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        print(out.shape)
        
        if self.in_channels == self.out_channels:
            out += identity
        else:
            out += self.projected(identity)
        out = self.relu2(out)
        
    def conv1x1(self, in_channels, out_channels, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        