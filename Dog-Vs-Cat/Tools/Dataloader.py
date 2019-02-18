import torch
import torchvision
import torchvision.transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()

class DriveData(Dataset):
    __xs = []
    __ys = []

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        
        for i in range(1,5000):
            address = folder_dataset + (str(i)+".jpg")
            self.__xs.append(address)

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        # Adding 
        if self.transform is not None:
            img = self.transform(img)
        else: 
            img = np.transpose(np.asarray(img), (2, 0,1))
            
        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        
        return img, 0

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

def datasets_loader(train_val_transforms=None, test_transforms=None):

    normalize = torchvision.transforms.Normalize((0.4895832, 0.4546405, 0.41594946), 
                                    (0.2520022, 0.24522494, 0.24728711))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                normalize])

    if train_val_transforms==None:
        train_val_transforms = transforms
    if test_transforms==None:
        test_transforms = transforms


    data = torchvision.datasets.ImageFolder(
                        root='./data/trainset/',
                        transform=train_val_transforms
                        )

    dataset_ratio = np.array([95, 5])/100

    sizes = [int(x*len(data)) for x in dataset_ratio]
    sizes[0] += len(data) - sum(sizes)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=data, lengths=sizes)
    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=128,
                        num_workers=2,
                        shuffle=True
                        )

    valid_loader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=128,
                        num_workers=2,
                        shuffle=True
    )       

    test_data = DriveData("./data/testset/test/", test_transforms)
    test_loader = DataLoader(test_data,  batch_size=1, shuffle=False, num_workers=1)

    return [train_loader, valid_loader, test_loader]


