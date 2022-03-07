from utils.utils import timer, get_label_dist, get_unique_labels, get_pixel_ranges
from utils.log import Log
from modeling.models import Resnet18, Resnet50, LeNetModel
import torch
import torch.nn as nn
import numpy as np
import os
import time


class Modeling():
    """Modeling class to steer the modeling for each subset"""

    def __init__(self, visible_gpus, log_id=None):
        ##### Logging #####
        self.log = Log(log_id).log if log_id else None

        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus  # '1,2,3'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.n_classes = None
        self.metrics = {}
        self.metrics['train_loss'] = []
        # Hyper Parameters
        self.num_epochs = 30  # 30
        self.batch_size = 32  # 96
        self.learning_rate = 0.001
        self.shuffle_train_loader = True
        self.datalader_workers = 1 if self.device == torch.device('cuda') else 0

    def run_modeling(self, train_dataset, test_dataset, model_id):
        """Run model training for given train and test data"""

        # Get number of classes to predict
        self.n_classes = len(get_unique_labels(test_dataset))
        if self.log:
            self.log.info(f"Training Dataset length: {len(train_dataset)}, classes: {self.n_classes} - {get_label_dist(train_dataset)}")

        # Get data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_train_loader, num_workers=self.datalader_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        if self.log:
            self.log.info(f"Range Trainloader: {get_pixel_ranges(train_loader)} - Range Testloader: {get_pixel_ranges(test_loader)}")

        # Select Model
        assert model_id in ['lenet', 'resnet18', 'resnet50'], "Invalid model_id, options are 'lenet', 'resnet18', 'resnet50'"
        if model_id == 'lenet':
            self.model = LeNetModel(self.n_classes)
        if model_id == 'resnet18':
            self.model = Resnet18(self.n_classes).model
        if model_id == 'resnet50':
            self.model = Resnet50(self.n_classes).model

        if self.log:
            self.log.info(
                f"Hyperparameter: model={model_id}, num_epochs={self.num_epochs}, batch_size={self.batch_size}, learning_rate={self.learning_rate}, shuffle_train_loader={self.shuffle_train_loader}, datalader_workers={self.datalader_workers}")

        # Train Model
        self.train_model(train_loader)

        # Test Model
        accuracy = self.test_model(test_loader)

        return accuracy

    def train_model(self, train_loader):
        """Train model with forward and backward pass and return trained model"""

        # Set non-deterministic (Not used to not limit performance)
        # torch.manual_seed(0)

        self.model.to(self.device)
        self.model.train()

        if self.log:
            self.log.info("# Start Training #")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.metrics['train_loss'] = []
        n_total_steps = len(train_loader)
        tr_start = time.time()
        # Training Loop
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.long().to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % n_total_steps == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            self.metrics['train_loss'].append(loss.item())

        tr_end = time.time()
        if self.log:
            self.log.info(f"# Training Successful - took {timer(tr_start, tr_end)}#")

        return self.model

    def test_model(self, test_loader):
        """Test model on test data and return accuracy"""

        assert self.model, "No model found. Run modeling first."
        # Put model in evaluation mode
        self.model.eval()
        with torch.set_grad_enabled(False):
            n_correct = 0
            n_samples = 0
            # Predict images
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            if self.log:
                self.log.info(f'Accuracy of the Network on Testdata: {acc:.4f} %')

        return acc
