from utils.utils import change_ds_transform, timer, get_items_idx_of_max_segment, get_label_dist, get_unique_counts, get_random_idx, get_ds_targets, get_norm_subset_idx
from utils.log import Log
import math
import time
import numpy as np
import copy
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, Subset
from skimage.metrics import structural_similarity as ssim


class SSIMSampler():
    """SSIM Sampler class to steer the application of structural similarity sampling"""

    def __init__(self, src_dataset, trgt_dataset, log_id=None):
        ##### Logging #####
        self.log = Log(log_id).log if log_id else None
        if self.log:
            self.log.info("Initializing SSIM Sampler...")

        assert isinstance(src_dataset, Dataset) and isinstance(trgt_dataset, Dataset), "Datasets are not type torch.utils.data.Dataset"
        self.src_ds = src_dataset
        self.trgt_ds = trgt_dataset

        self.src_imgs_ssims = self.get_src_images_ssims(self.src_ds, self.trgt_ds)  # List of Lists

        self.src_imgs_ssim_avgs = self.get_ssim_avgs(self.src_imgs_ssims)
        self.src_imgs_ssim_maxs = self.get_ssim_maxima(self.src_imgs_ssims)
        self.src_imgs_ssim_kmax_avgs = self.get_ssim_kmax_avgs(self.src_imgs_ssims)

        if self.log:
            self.log.info("Finished initiating SSIM Sampler.")

    def get_src_images_ssims(self, src_ds, trgt_ds):
        """Return structural similarity beween image features.

        Returns two-dimensional array containing n-sized lists
        of similarity measures for each source images - here 
        the SSIM.

        Args: 
            src_ds (torch.utils.data.Dataset): Source dataset
            src_ds (torch.utils.data.Dataset): Target dataset
        
        Returns:
            numpy.ndarray: list of lists with similarity measures for each image
        """

        # Take random n images from target for comparison
        n = 500
        r_idx = get_random_idx(trgt_ds, n)  # Sample idx
        trgt_compare_ds = Subset(trgt_ds, r_idx)  # Create Subset Dataset
        trgt_label_dist = get_label_dist(trgt_compare_ds)  # Get Label Distribution

        if self.log:
            self.log.info(f"SSIM: Comparing Source (={len(src_ds)}) to Target (={len(trgt_compare_ds)}): {trgt_label_dist}")

        # Temporarly change Transformation
        src_ds_copy = copy.deepcopy(src_ds)
        trgt_compare_ds_copy = copy.deepcopy(trgt_compare_ds)

        ssim_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        src_ds_copy = change_ds_transform(src_ds_copy, ssim_transform)
        trgt_compare_ds_copy = change_ds_transform(trgt_compare_ds_copy, ssim_transform)

        # Into Np Array for skimage ssim compatability
        src_array = np.array([img[0][0].numpy() for img in src_ds_copy])
        trgt_array = np.array([img[0][0].numpy() for img in trgt_compare_ds_copy])
        # One ssim-list for each image in length of trgt dataset
        src_imgs_ssims = []
        tstart = time.time()
        # For each image, get list of ssim to all trgt_imgs
        for i, s_img in enumerate(src_array):
            s_img_ssims = [ssim(s_img, t_img) for t_img in trgt_array]
            src_imgs_ssims.append(s_img_ssims)

            if i == 0 or (i+1) % int(math.ceil(len(src_ds_copy)/10)) == 0 or (i+1) % len(src_ds_copy) == 0:
                tend = time.time()
                if self.log:
                    self.log.info(f"Computed SSIM for {i+1} / {len(self.src_ds)} - took {timer(tstart, tend)}")

        src_imgs_ssims = np.array(src_imgs_ssims)
        return src_imgs_ssims

    def get_ssim_avgs(self, imgs_ssims):
        """Take average ssim for each image"""
        ssim_avg_list = np.array([sum(ssims_list)/len(ssims_list) for ssims_list in imgs_ssims])
        return ssim_avg_list

    def get_ssim_maxima(self, imgs_ssims):
        """Take maximum ssim for each image"""
        ssim_max_list = np.array([max(ssims_list) for ssims_list in imgs_ssims])
        return ssim_max_list

    def get_ssim_kmax_avgs(self, imgs_ssims, k=5):
        """Take maximum ssim for each image"""
        imgs_ssims_sorted = [sorted(ssims_list, reverse=True) for ssims_list in imgs_ssims]
        ssim_k_max_avg = np.array([sum(ssims_list[:k])/k for ssims_list in imgs_ssims_sorted])
        return ssim_k_max_avg

    def get_max_ssim_subset(self, dataset, subset_size, mode_id=None, class_equal=False, ds_origins=None):
        """Get subset according to similarity - here distance measure SSIM and mode"""
        assert mode_id in ['ssim_avg', 'ssim_max', 'ssim_k_max'], "Invalid mode_id, options are 'ssim_avg', 'ssim_max', 'ssim_k_max'"
        if mode_id == 'ssim_avg':
            distance_measure_list = self.src_imgs_ssim_avgs
        elif mode_id == 'ssim_max':
            distance_measure_list = self.src_imgs_ssim_maxs
        elif mode_id == 'ssim_k_max':
            distance_measure_list = self.src_imgs_ssim_kmax_avgs

        assert len(dataset) == len(distance_measure_list), "Dataset and SSIMs array don't have the same length or src_imgs_ssims is None."

        if class_equal:
            # Get idx_max_segment sampling all classes as equally as possible
            idx_max_segment = get_norm_subset_idx(dataset, subset_size, distance_measure_list, segment='max')
        else:
            # Get idx_max_segment according to overall ranking
            idx_max_segment = get_items_idx_of_max_segment(subset_size, distance_measure_list)

        # Origin Dataset Distribution
        origin_dist = get_unique_counts(ds_origins[idx_max_segment])
        if self.log:
            self.log.info(f"Orgin Dataset Distribution: {origin_dist}")
        # Ceate Subset
        subset_ds = Subset(dataset, idx_max_segment)

        return subset_ds
