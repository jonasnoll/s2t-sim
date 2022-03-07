from modeling.models import BaseAutoencoder
from utils.log import Log
from utils.utils import timer
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import time


class Autoencoder():
    def __init__(self, visible_gpus, log_id, img_channels=1):
        ##### Logging #####
        self.log = Log(log_id).log if log_id else None

        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus  # '1,2,3'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert img_channels in [1, 3], "Caution: img_channels should be either int 1 or 3"
        self.img_channels = img_channels

        # Model
        self.model = None
        self.metrics = {}
        self.metrics['train_loss'] = []
        self.outputs = []

        # Hyper Parameters
        self.num_epochs = 8 if img_channels == 1 else 120  # Train longer for more complex DomainNet data
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.shuffle_train_loader = True
        self.datalader_workers = 1
        if self.log:
            self.log.info(
                f"AE Hyperparameter: num_epochs={self.num_epochs}, batch_size={self.batch_size}, learning_rate={self.learning_rate}, shuffle_train_loader={self.shuffle_train_loader}, datalader_workers={self.datalader_workers}, (img_channels={self.img_channels})")

    def train_autoencoder(self, dataset):
        """Run the autoencoder training"""

        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle_train_loader)

        # Init base autoencoder
        self.model = BaseAutoencoder(input_channels=self.img_channels)  # 3
        self.model.to(self.device)

        if self.log:
            self.log.info("# Start AE Training #")
        tr_start = time.time()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        # Training Loop
        for epoch in range(self.num_epochs):
            for i, (img, _) in enumerate(data_loader):
                img = img.to(self.device)
                
                # Forward pass
                recon = self.model(img)
                loss = criterion(recon, img)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.log:
                self.log.info(f'Epoch: {epoch+1}/{self.num_epochs}, Loss:{loss.item():.4f}')
            self.metrics['train_loss'].append(loss.item())
            self.outputs.append((epoch, img.cpu(), recon.cpu()))

        tr_end = time.time()
        if self.log:
            self.log.info(f"# AE Training Successful - took {timer(tr_start, tr_end)}#")

    def get_recon_losses(self, dataset, save_imgs=True):
        """Run reconstruction on dataset (source) and return the loss scores for all samples"""

        if self.log:
            self.log.info(f"Predicting Recon Losses ({len(dataset)})...")

        # Use batch size 1 so loss is calculated for individual images
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        self.model.eval()
        criterion = nn.MSELoss()

        losses = []
        imgs = []
        recons = []
        with torch.set_grad_enabled(False):
            for i, (img, _) in enumerate(data_loader):
                img = img.to(self.device)
                recon = self.model(img)
                loss = criterion(recon, img)

                losses.append(loss.item())
                # Save 32 imaegs for analysis
                if i < 32:
                    imgs.append(img)
                    recons.append(recon)

        if save_imgs:
            path = './results/img'
            save_image(torch.cat(imgs), f'{path}/32_autoencder_original.png')
            save_image(torch.cat(recons), f'{path}/32_autoencder_recon.png')

        return np.array(losses)
