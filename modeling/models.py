import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class LeNetModel(nn.Module):
    def __init__(self, num_classes):
        super(LeNetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Resnet18:
    def __init__(self, num_classes):
        self.n_classes = num_classes
        self.model = self.resnet()

    def resnet(self):
        model = torchvision.models.resnet18(pretrained=True)
        # Freeze Model
        for param in model.parameters():
            param.requires_grad = False

        # Add unfrozen fully conected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.n_classes)

        return model


class Resnet50:
    def __init__(self, num_classes):
        self.n_classes = num_classes
        self.model = self.resnet()

    def resnet(self):
        model = torchvision.models.resnet50(pretrained=True)
        # Freeze Model
        for param in model.parameters():
            param.requires_grad = False

        # Add unfrozen fully conected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.n_classes)

        return model


class BaseAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # N, 1 or 3, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),  # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # -> N, 64, 1, 1
        )
        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, 3, stride=2, padding=1, output_padding=1),  # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()  # Final value between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
