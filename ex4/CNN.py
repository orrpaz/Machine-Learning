import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,image_size):
        super(CNN, self).__init__()
        self.image_size = image_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(image_size, 1000)
        self.fc2 = nn.Linear(1000, 30)

class CNN2(nn.Module):
    def __init__(self, image_size):
        super(CNN2, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, stride=1)
        self.drop_out = nn.Dropout()
        self.fc0 = nn.Linear(12*37*22, 1000)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 30)



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # x = x.view(-1, 12*37*22)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)



class MyCNN(nn.Module):
    def __init__(self, image_size):
        super(MyCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 15, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(15, 32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(32, 20, kernel_size=5, stride=1)
        self.drop_out = nn.Dropout()
        self.fc0 = nn.Linear(20*16*9, 1000)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 30)




    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 20*16*9)
        # out_re = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


