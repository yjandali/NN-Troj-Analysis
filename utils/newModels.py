import torch 
import torch.nn as nn 
import torch.nn.functional as F

fcSizes = 512*2
hiddenSize1 = 8092
hiddenSize2 = 4096
num_classes = 2

class SimplerDenseNNfc(nn.Module):
    def __init__(self):
        super(SimplerDenseNNfc, self).__init__()
        self.fc1 = nn.Linear(fcSizes, hiddenSize1)
        self.fc2 = nn.Linear(hiddenSize1, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, fcSizes)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x