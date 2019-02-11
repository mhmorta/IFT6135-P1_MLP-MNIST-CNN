

import torch
import torch.nn as nn
from Modules.Dropout import dropout

class XavierClassifier(nn.Module):
    def __init__(self):
        super(XavierClassifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=16,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(0.1, 16,'gaussian'),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            dropout(0.1, 32,'gaussian'),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(0.1, 64,'gaussian'),
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
            dropout(0.1, 256,'gaussian'),
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
            dropout(0.1, 512,'gaussian'),
            
        )
        self.net = nn.Linear(512, 512)
        self.net = nn.Linear(512, 2)
        self.init_weights()
    def forward(self, x):
        return self.net(self.conv(x).squeeze())
    
    def init_weights(self):
        for m in self.conv:
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if type(self.net) == nn.Linear:
            nn.init.xavier_uniform_(self.net.weight)
            self.net.bias.data.fill_(0.01)


class XavierDropout(nn.Module):
    def __init__(self, hyperparams):
        super(XavierDropout, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=16,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(hyperparams['dropout'], 16,'gaussian'),
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            dropout(hyperparams['dropout'], 16,'gaussian'),
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(hyperparams['dropout'], 16,'gaussian'),
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(hyperparams['dropout'], 16,'gaussian'),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # Layer 5
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(hyperparams['dropout'], 16,'gaussian'),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # Layer 6
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            dropout(hyperparams['dropout'], 16,'gaussian'),
             # Layer 7
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(hyperparams['dropout'], 16,'gaussian'),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # Layer 8
            nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(hyperparams['dropout'], 16,'gaussian'),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # Layer 9
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)          
        )
        self.net = nn.Linear(512, 512)
        self.net = nn.Linear(512, 2)
        self.init_weights()
    def forward(self, x):
        return self.net(self.conv(x).squeeze())
    
    def init_weights(self):
        for m in self.conv:
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if type(self.net) == nn.Linear:
            nn.init.xavier_uniform_(self.net.weight)
            self.net.bias.data.fill_(0.01)

