from utils.utils import change_ds_transform, get_label_dist, get_random_idx, get_unique_counts, timer
from sampler.ranking import get_items_idx_of_min_segment, get_norm_subset_idx
from utils.log import Log
import torch
from torch.utils.data import Dataset, Subset
from modeling.feature_extractor import FeatureExtractor
import math
import time
from scipy import spatial
import copy
import numpy as np
import torchvision.transforms as transforms


class FeatureDistSampler(): 
    """Feature Sampler class to steer the application of learned feature space distance sampling"""

    def __init__(self, src_dataset, trgt_dataset, dist='cos', visible_gpus='0', log_id=None):
        ##### Logging #####
        self.log = Log(log_id).log if log_id else None
        if self.log:
            self.log.info("Initializing Dist Sampler...")
        self.log_id = log_id
        self.visible_gpus = visible_gpus

        assert isinstance(src_dataset, Dataset) and isinstance(trgt_dataset, Dataset), "Datasets are not type torch.utils.data.Dataset"
        self.src_ds = src_dataset
        self.trgt_ds = trgt_dataset

        self.dist_measure = dist  # from 'cos' or 'euclid'
        self.src_imgs_dists = self.get_src_feat_dists(self.src_ds, self.trgt_ds)  # List of Lists

        self.src_imgs_dist_avgs = self.get_dist_avgs(self.src_imgs_dists)
        self.src_imgs_dist_mins = self.get_dist_minima(self.src_imgs_dists)
        self.src_imgs_dist_kmin_avgs = self.get_dist_kmin_avgs(self.src_imgs_dists)

        if self.log:
            self.log.info("Finished initiating Dist Sampler.")

    def get_features(self, dataset):
        """Extract features from images with feature extractor."""
        # Temporarly change Transformation
        ds_copy = copy.deepcopy(dataset)

        resnet_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        ds_copy = change_ds_transform(ds_copy, resnet_transform)

        fe = FeatureExtractor(self.visible_gpus, self.log_id)
        ftrs = fe.get_features(ds_copy)

        return ftrs

    def get_src_feat_dists(self, src_ds, trgt_ds):
        """Return disttances beween image features.

        Returns two-dimensional array containing n-sized lists
        of similarity measures for each source images - here 
        the distance between extracted features.

        Args: 
            src_ds (torch.utils.data.Dataset): Source dataset
            src_ds (torch.utils.data.Dataset): Target dataset
        
        Returns:
            numpy.ndarray: list of lists with similarity measures for each image
        """

        # Take random n images from target for comparison
        n = 500
        r_idx = get_random_idx(self.trgt_ds, n)  # Sample idx
        trgt_compare_ds = Subset(trgt_ds, r_idx)  # Create Subset Dataset
        trgt_label_dist = get_label_dist(trgt_compare_ds)  # Get Label Distribution

        if self.log:
            self.log.info(f"FTR: Comparing Source (={len(src_ds)}) to Target (={len(trgt_compare_ds)}): {trgt_label_dist}")

        # Get Features
        src_ftrs = self.get_features(src_ds)
        trgt_ftrs = self.get_features(trgt_compare_ds)  # len of n

        src_img_dists = []
        tstart = time.time()
        # For each image, get list of ssim to all trgt_imgs
        for i, s_ftr_vec in enumerate(src_ftrs):
            curr_dists = []
            for t_ftr_vec in trgt_ftrs:
                if self.dist_measure == 'euclid':
                    # Eucledean Distance
                    dist = np.linalg.norm(s_ftr_vec - t_ftr_vec)
                    curr_dists.append(dist)
                if self.dist_measure == 'cos':
                    # Cosine Distance
                    dist = spatial.distance.cosine(s_ftr_vec, t_ftr_vec)
                    curr_dists.append(dist)

            src_img_dists.append(curr_dists)

            if i == 0 or (i+1) % int(math.ceil(len(src_ftrs)/10)) == 0 or (i+1) % len(src_ftrs) == 0:
                tend = time.time()
                if self.log:
                    self.log.info(f"Computed Feature Dist for {i+1} / {len(src_ftrs)} - took {timer(tstart, tend)}")

        return np.array(src_img_dists)

    def get_dist_avgs(self, imgs_dists):
        """Take average dist for each image"""
        dist_avg_list = np.array([sum(dists_list)/len(dists_list) for dists_list in imgs_dists])
        return dist_avg_list

    def get_dist_minima(self, imgs_dists):
        """Take minima dist for each image"""
        dist_min_list = np.array([min(dists_list) for dists_list in imgs_dists])
        return dist_min_list

    def get_dist_kmin_avgs(self, imgs_dists, k=5):
        """Take avg min dist for 5 dists for each image"""
        imgs_dists_sorted = [sorted(dists_list, reverse=False) for dists_list in imgs_dists]
        dist_k_min_avgs = np.array([sum(dists_list[:k])/k for dists_list in imgs_dists_sorted])
        return dist_k_min_avgs

    def get_min_dist_subset(self, dataset, subset_size, mode_id=None, class_equal=False, ds_origins=None):
        """Get subset according to similarity - here distance measure feature distance and mode"""
        assert mode_id in ['dist_avg', 'dist_min', 'dist_k_min'], "Invalid mode_id, options are 'dist_avg', 'dist_min', 'dist_k_min'"
        if mode_id == 'dist_avg':
            distance_measure_list = self.src_imgs_dist_avgs
        elif mode_id == 'dist_min':
            distance_measure_list = self.src_imgs_dist_mins
        elif mode_id == 'dist_k_min':
            distance_measure_list = self.src_imgs_dist_kmin_avgs

        assert (len(dataset) == len(distance_measure_list)), "Dataset and Feature Distance Array don't have the same length."

        if class_equal:
            # Get idx_min_segment sampling all classes as equally as possible: Min distane, means high similarity
            idx_min_segment = get_norm_subset_idx(dataset, subset_size, distance_measure_list, segment='min')
        else:
            # Get idx_min_segment according to overall ranking
            idx_min_segment = get_items_idx_of_min_segment(subset_size, distance_measure_list)

        # Origin Dataset Distribution
        origin_dist = get_unique_counts(ds_origins[idx_min_segment])
        if self.log:
            self.log.info(f"Orgin Dataset Distribution: {origin_dist}")
        # Ceate Subset
        subset_ds = Subset(dataset, idx_min_segment)

        return subset_ds
