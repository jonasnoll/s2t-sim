from utils.log import Log
from modeling.autoencoder import Autoencoder
from utils.utils import get_items_idx_of_min_segment, change_ds_transform, get_unique_counts, get_norm_subset_idx
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import copy


class AutoencoderSampler():
    """Autoencoder Sampler class to steer the application of the autoencoder loss sampling"""

    def __init__(self, src_dataset, trgt_dataset, img_channels=1, visible_gpus='0', log_id=None):
        ##### Logging #####
        self.log_id = log_id
        self.log = Log(log_id).log if log_id else None
        if self.log:
            self.log.info("Initializing Autoencoder Sampler...")

        self.visible_gpus = visible_gpus

        assert img_channels in [1, 3], "Caution: img_channels should be either int 1 or 3"
        self.img_channels = img_channels

        assert isinstance(src_dataset, Dataset) and isinstance(trgt_dataset, Dataset), "Datasets are not type torch.utils.data.Dataset"
        self.src_ds = src_dataset
        self.trgt_ds = trgt_dataset

        self.autoencoder = None
        self.src_img_losses = self.get_src_img_losses(self.src_ds, self.trgt_ds, self.img_channels)

    def get_src_img_losses(self, src_ds, trgt_ds, img_channels):
        """Returns list of autoencoder loss for each image used as the similarity measure"""
        # Transform Data occording to input channels
        autoencoder_transform = {
            '1': transforms.Compose([
                transforms.Resize([28, 28]),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]),
            '3': transforms.Compose([
                transforms.Resize([28, 28]),
                transforms.ToTensor(),
            ]),
        }
        src_ds_copy = copy.deepcopy(src_ds)
        trgt_ds_copy = copy.deepcopy(trgt_ds)

        src_ds_copy = change_ds_transform(src_ds_copy, autoencoder_transform[str(img_channels)])
        trgt_ds_copy = change_ds_transform(trgt_ds_copy, autoencoder_transform[str(img_channels)])

        # Train Autoencoder
        self.autoencoder = Autoencoder(self.visible_gpus, self.log_id, img_channels)
        self.autoencoder.train_autoencoder(trgt_ds_copy)

        # Get Losses
        losses = self.autoencoder.get_recon_losses(src_ds_copy)

        return losses

    def get_min_autoencoder_subset(self, dataset, subset_size, class_equal=False, ds_origins=None):
        """Get subset according to similarity - here distance measure autoencoder loss and mode"""
        distance_measure_list = self.src_img_losses
        assert len(dataset) == len(distance_measure_list), "Dataset and Autoencoder Recon Loss Array don't have the same length or src_imgs_ssims is None."

        if class_equal:
            # Get idx_max_segment sampling all classes as equally as possible
            idx_min_segment = get_norm_subset_idx(dataset, subset_size, distance_measure_list, segment='min')
        else:
            # Get idx_max_segment according to overall ranking
            idx_min_segment = get_items_idx_of_min_segment(subset_size, distance_measure_list)

        # Origin Dataset Distribution
        origin_dist = get_unique_counts(ds_origins[idx_min_segment])
        if self.log:
            self.log.info(f"Orgin Dataset Distribution: {origin_dist}")
        # Ceate Subset
        subset_ds = Subset(dataset, idx_min_segment)

        return subset_ds
