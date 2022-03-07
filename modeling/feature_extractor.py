from utils.log import Log
from modeling.models import Resnet18
import os
import numpy as np
import torch
import torch.nn as nn


class FeatureExtractor:
    def __init__(self, visible_gpus, log_id):
        ##### Logging #####
        self.log = Log(log_id).log if log_id else None

        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus  # '1,2,3'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_ftrs = 0
        self.headless_model = self.get_headless()

    def get_headless(self):
        """Returns headless model for extracting features with"""
        headless_model = Resnet18(10).model
        for param in headless_model.parameters():
            param.requires_grad = False

        self.num_ftrs = headless_model.fc.in_features
        if self.log:
            self.log.info(f"Extracted features should be size: {self.num_ftrs}")

        # Set fully connected to "Identity"placeholder
        headless_model.fc = nn.Identity()

        return headless_model

    def get_features(self, dataset):
        """Running Feature extraction and returning list of image features"""
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)  # 256
        self.headless_model.to(self.device)
        self.headless_model.eval()

        with torch.no_grad():
            dataset_features = torch.Tensor().to(self.device)
            for images, labels in data_loader:
                images = images.to(self.device)

                outputs = self.headless_model(images)
                assert self.num_ftrs == outputs.size(1), f"Outputs are not {self.num_ftrs}: size {outputs.size(1)}"

                dataset_features = torch.cat((dataset_features, outputs), 0)

        # Return tensor to cpu
        dataset_features = dataset_features.cpu()
        dataset_features = dataset_features.numpy()

        if self.log:
            self.log.info(f"Returning Features: ({dataset_features.shape})")

        return dataset_features
