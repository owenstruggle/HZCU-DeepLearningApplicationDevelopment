import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class FashionMnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TwoConv2dNet(BaseModel):
    def __init__(self, n=0.3):
        super(TwoConv2dNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(n),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(n)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x


class FourConv2dNet(BaseModel):
    def __init__(self, n=0.3):
        super(FourConv2dNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 1, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(n),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(n),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(n),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)
        return x


class SixConv2dNet(BaseModel):
    def __init__(self, n=0.3):
        super(SixConv2dNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 1, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(n),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(n),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(n),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(n),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.fc(x)
        return x


class EightConv2dNet(BaseModel):
    def __init__(self, n=0.3):
        super(EightConv2dNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 1, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(n),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(n),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(n),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(n),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(n),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512 * 3 * 3)
        x = self.fc(x)
        return x
