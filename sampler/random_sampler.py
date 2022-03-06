from torch.utils.data import Dataset, Subset
from utils.utils import get_unique_counts
from utils.log import Log
import torch
import numpy as np
import random


class RandomSampler():
    def __init__(self, log_id=None):
        ##### Logging #####
        self.log = Log(log_id).log if log_id else None
        # # this.input_data = None

    def get_random_subset(self, input_ds, subset_size, ds_origins=None):
        if not isinstance(input_ds, Dataset):
            raise TypeError("input_ds is not type torch.utils.data.Dataset")

        # Set Amount to Sample
        sampling_amount = int(len(input_ds) * subset_size)
        if sampling_amount == 0:
            sampling_amount = 1
        # Set pseudo idx list
        idx = list(range(len(input_ds)))
        # Sample idx
        r_idx = random.sample(idx, sampling_amount)
        r_idx.sort()

        # Ceate Subset
        subset_ds = Subset(input_ds, r_idx)
        if self.log:
            self.log.info(
                f"Random Subet for {subset_size:<4} of {len(input_ds)} imgs -> {len(subset_ds)} imgs")

        # Print Origin Dataset Distribution
        try:
            origin_dist = get_unique_counts(ds_origins[r_idx])
            print(f"Orgin Dataset Distribution: {origin_dist}")
            if self.log:
                self.log.info(f"Orgin Dataset Distribution: {origin_dist}")
        except:
            print("No Orgin Dataset Distribution available")

        return subset_ds
