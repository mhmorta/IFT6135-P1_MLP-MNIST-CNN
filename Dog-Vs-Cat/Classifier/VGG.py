
import torch
import torch.nn as nn
from Modules.Dropout import dropout


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv = nn.Sequential(
            # Layaer 1
            nn.Conv2d(in_channels=3, out_channels=64,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            dropout(0.25, 64,'gaussian'),

            nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            
            # Layaer 3
            nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            
            # Layaer 5
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3,3) , padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            
             # Layaer 8
            nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
        
            # Layaer 11
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            
            # Layaer 12
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            
            # Layaer 13
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(3,3) , padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),  
            dropout(0.5, 512,'gaussian'),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2*2*512, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 2),
        )
        self.init_weights()
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 2*2*512)
        x = self.classifier(x)
        return x

    def init_weights(self):
        for m in self.conv:
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if type(self.classifier) == nn.Linear:
            nn.init.xavier_uniform_(self.classifier.weight)
            self.classifier.bias.data.fill_(0.01)
