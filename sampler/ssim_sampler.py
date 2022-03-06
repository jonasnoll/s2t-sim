# from utils.utils import Utils
from utils.utils import change_ds_transform, timer, get_items_idx_of_max_segment, get_label_dist, get_unique_counts, get_random_idx, get_ds_targets, get_norm_subset_idx
from utils.log import Log
import torch
from torch.utils.data import Dataset, Subset
from skimage.metrics import structural_similarity as ssim

import math
import time
import numpy as np
import copy
import torchvision.transforms as transforms


class SSIMSampler():
    def __init__(self, src_dataset, trgt_dataset, log_id=None):
        ##### Logging #####
        self.log = Log(log_id).log if log_id else None
        print("Initializing SSIM Sampler...")
        if self.log:
            self.log.info("Initializing SSIM Sampler...")

        assert isinstance(src_dataset, Dataset) and isinstance(trgt_dataset, Dataset), "Datasets are not type torch.utils.data.Dataset"
        self.src_ds = src_dataset
        self.trgt_ds = trgt_dataset

        # Selecting 1% of Target Data for comparison
        # self.amount_trgt_comparisons = int(0.01 * len(self.trgt_ds))
        # print(f"Comparing each image to {self.amount_trgt_comparisons} (=1%) of the target data (={len(self.trgt_ds)})")

        self.src_imgs_ssims = self.get_src_images_ssims(self.src_ds, self.trgt_ds)  # List of Lists

        self.src_imgs_ssim_avgs = self.get_ssim_avgs(self.src_imgs_ssims)
        self.src_imgs_ssim_maxs = self.get_ssim_maxima(self.src_imgs_ssims)
        self.src_imgs_ssim_kmax_avgs = self.get_ssim_kmax_avgs(self.src_imgs_ssims)

        print("Finished initiating SSIM Sampler.")
        if self.log:
            self.log.info("Finished initiating SSIM Sampler.")

    def get_src_images_ssims(self, src_ds, trgt_ds):
        print("Got original shapes:")
        print(type(src_ds))
        print(type(src_ds[0]))
        print(type(src_ds[0][0]))
        print(f"src_ds: {src_ds[0][0].shape}")
        print(f"trgt_ds: {trgt_ds[0][0].shape}")

        # For Testing Purposes, take random n images from target for comparison
        n = 500  # 20
        r_idx = get_random_idx(trgt_ds, n)  # Sample idx
        trgt_compare_ds = Subset(trgt_ds, r_idx)
        trgt_label_dist = get_label_dist(trgt_compare_ds)

        if self.log:
            self.log.info(f"SSIM: Comparing Source (={len(src_ds)}) to Target (={len(trgt_compare_ds)}): {trgt_label_dist}")

        # Change Transformation
        src_ds_copy = copy.deepcopy(src_ds)
        trgt_compare_ds_copy = copy.deepcopy(trgt_compare_ds)

        ssim_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, ), (0.5, )),
        ])

        src_ds_copy = change_ds_transform(src_ds_copy, ssim_transform)
        trgt_compare_ds_copy = change_ds_transform(trgt_compare_ds_copy, ssim_transform)

        print("Sampling with transformed shapes:")
        print(f"src_ds_copy: {src_ds_copy[0][0].shape}")
        print(f"trgt_ds_copy: {trgt_compare_ds_copy[0][0].shape}")

        # Into Np Array for skimage ssim compatability
        src_array = np.array([img[0][0].numpy() for img in src_ds_copy])
        trgt_array = np.array([img[0][0].numpy() for img in trgt_compare_ds_copy])
        # One ssim-list for each image in length of trgt dataset
        src_imgs_ssims = []
        tstart = time.time()

        # # For Testing Purposes, take random n images from target for comparison
        # n = 1000 # 20
        # r_idx = get_random_idx(trgt_ds, n) # Sample idx
        # trgt_label_dist = get_unique_counts(get_ds_targets(trgt_ds)[r_idx])
        # if self.log: self.log.info(f"SSIM: Comparing Source to Target Dist (len={len(trgt_array[r_idx])}): {trgt_label_dist}")

        for i, s_img in enumerate(src_array):
            # For each image, get list of ssim to all trgt_imgs
            # s_img_ssims = [ssim(s_img, t_img) for t_img in trgt_array[:self.amount_trgt_comparisons]]
            # s_img_ssims = [ssim(s_img, t_img) for t_img in trgt_array[r_idx]]
            s_img_ssims = [ssim(s_img, t_img) for t_img in trgt_array]

            src_imgs_ssims.append(s_img_ssims)

            if i == 0 or (i+1) % int(math.ceil(len(src_ds_copy)/10)) == 0 or (i+1) % len(src_ds_copy) == 0:
                tend = time.time()
                print(f"Computed SSIM for {i+1} / {len(self.src_ds)} - took {timer(tstart, tend)}")
                if self.log:
                    self.log.info(f"Computed SSIM for {i+1} / {len(self.src_ds)} - took {timer(tstart, tend)}")

        src_imgs_ssims = np.array(src_imgs_ssims)
        return src_imgs_ssims

    def get_ssim_avgs(self, imgs_ssims):
        # Take average ssim for each image
        ssim_avg_list = np.array([sum(ssims_list)/len(ssims_list) for ssims_list in imgs_ssims])
        return ssim_avg_list

    def get_ssim_maxima(self, imgs_ssims):
        # Take maximum ssim for each image
        ssim_max_list = np.array([max(ssims_list) for ssims_list in imgs_ssims])
        return ssim_max_list

    def get_ssim_kmax_avgs(self, imgs_ssims, k=5):
        # Take maximum ssim for each image
        imgs_ssims_sorted = [sorted(ssims_list, reverse=True) for ssims_list in imgs_ssims]
        ssim_k_max_avg = np.array([sum(ssims_list[:k])/k for ssims_list in imgs_ssims_sorted])
        return ssim_k_max_avg

    def get_max_ssim_subset(self, dataset, subset_size, mode_id=None, class_equal=False, ds_origins=None):
        # Get distance measure according to mode
        if mode_id == 'ssim_avg':
            distance_measure_list = self.src_imgs_ssim_avgs
        elif mode_id == 'ssim_max':
            distance_measure_list = self.src_imgs_ssim_maxs
        elif mode_id == 'ssim_k_max':
            distance_measure_list = self.src_imgs_ssim_kmax_avgs
        else:
            raise ValueError("Wrong or missing mode_id, options are 'ssim_avg', 'ssim_max', 'ssim_k_max'")

        assert len(dataset) == len(distance_measure_list), "Datset and SSIMs Array don't have the same length or src_imgs_ssims is None."

        if class_equal:
            # Get idx_max_segment sampling all classes as equally as possible
            idx_max_segment = get_norm_subset_idx(dataset, subset_size, distance_measure_list, segment='max')
        else:
            # Get idx_max_segment according to overall ranking
            idx_max_segment = get_items_idx_of_max_segment(subset_size, distance_measure_list)

        # Print Origin Dataset Distribution
        origin_dist = get_unique_counts(ds_origins[idx_max_segment])
        print(f"Orgin Dataset Distribution: {origin_dist}")
        if self.log:
            self.log.info(f"Orgin Dataset Distribution: {origin_dist}")
        # Ceate Subset
        subset_ds = Subset(dataset, idx_max_segment)

        print(f"Returning SSIM Sampler Subset with length {len(subset_ds)} and data shape {subset_ds[0][0].shape}")
        print(f"And Label count:")
        print(get_label_dist(subset_ds))
        print("And Shape:")
        print(f"subset_ds: {subset_ds[0][0].shape}")

        print(f"Lowest Subset: {min(distance_measure_list[idx_max_segment])}")
        print(f"Highest Subset: {max(distance_measure_list[idx_max_segment])}")

        return subset_ds

    # def get_max_ssim_subset(self, dataset, subset_size, mode_id=None):
    #     # TODO: Hier als Argument noch class_equal rein und dan in sef.get_subset übergeben
    #     if mode_id == 'ssim_avg':
    #         return self.get_subset(dataset, subset_size, self.src_imgs_ssim_avgs)
    #     if mode_id == 'ssim_max':
    #         return self.get_subset(dataset, subset_size, self.src_imgs_ssim_maxs)
    #     if mode_id == 'ssim_k_max':
    #         return self.get_subset(dataset, subset_size, self.src_imgs_ssim_kmax_avgs)
    #     else:
    #         raise ValueError("Wrong or missing mode_id, options are 'ssim_avg', 'ssim_max', 'ssim_k_max'")
