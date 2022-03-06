import json
import time
import numpy as np
# import cv2
import os
import scipy.io
import torch
import torchvision
import gzip
import h5py
import glob
import sys
import logging
from random import Random
# from matplotlib import pyplot as plt
from utils.datasets import Digit5Subset


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}"


def create_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)
        else:
            print("Successfully created the directory %s" % dir_path)
    else:
        print("Seems like path %s exists." % dir_path)


def get_pixel_ranges(dataloader):
    if isinstance(dataloader, torch.utils.data.dataset.Dataset):
        dataloader = torch.utils.data.DataLoader(
            dataloader, batch_size=128, shuffle=False, num_workers=0)

    assert isinstance(
        dataloader, torch.utils.data.DataLoader), "Argument dataloader should be type torch.utils.data.dataset.DataLoader"
    dataiter = iter(dataloader)
    imgs, labels = dataiter.next()
    pix_range = (torch.min(imgs), torch.max(imgs))

    return pix_range


def combine_datasets(datasets):
    combined_dataset = np.concatenate(datasets, axis=0)

    return combined_dataset


# def plot_histogram(distribution, title="Histogram", print_frequencies=True):
#     (unique, counts) = np.unique(distribution, return_counts=True)
#     frequencies = np.asarray((unique, counts)).T
#     if print_frequencies:
#         for label, freq in frequencies:
#             print(f"{label} : {freq}")

#     plt.hist(distribution, bins=[0, 1, 2, 3, 4, 5,
#                                  6, 7, 8, 9, 10], rwidth=0.5, align='left')
#     plt.title(title)
#     plt.show()


# def plot_dists(distributions):
#     fig = plt.figure(figsize=(4, 2))
#     ax = fig.add_axes([0,0,1,1])
#     categories = [str(i) for i in range(10)]
#     ax.bar(categories, distributions)
#     plt.show()

# def plot_dists(distributions):
#     x = []
#     y = []
#     for k, v in distributions.items():
#         x.append(str(k))
#         y.append(int(v))

#     plt.bar(x, y)
#     plt.show()


def save_as_txt(data, filepath):
    with open(filepath, "w") as text_file:
        text_file.write(data)


def save_as_json(data, filepath):
    # print("Saving as ", filepath)
    with open(filepath, 'w') as fp:
        json.dump(data, fp, indent=4)


def get_items_idx_of_max_segment(segment_size, distance_measure_list):
    """Returns a list of indices for the highest values within a list."""
    amount_to_take = int(len(distance_measure_list)*segment_size)
    try:
        if amount_to_take >= len(distance_measure_list):
            return np.array(range(len(distance_measure_list)))
        else:
            if amount_to_take == 0:
                amount_to_take = 1
            idx_max_segment = np.argpartition(
                distance_measure_list, -amount_to_take)[-amount_to_take:]
            return idx_max_segment
    except:
        return np.array(range(len(distance_measure_list)))


def get_n_idx_of_max_segment(n, distance_measure_list):
    amount_to_take = n
    try:
        if amount_to_take >= len(distance_measure_list):
            idx_max_segment = np.array(range(len(distance_measure_list)))
        else:
            if amount_to_take == 0:
                amount_to_take = 1
            idx_max_segment = np.argpartition(
                distance_measure_list, -amount_to_take)[-amount_to_take:]
    except:
        idx_max_segment = np.array(range(len(distance_measure_list)))

    return idx_max_segment


def get_items_idx_of_min_segment(segment_size, distance_measure_list):
    """Returns a list of indices for the lowest values within a list."""
    amount_to_take = int(len(distance_measure_list)*segment_size)
    try:
        if amount_to_take >= len(distance_measure_list):
            return np.array(range(len(distance_measure_list)))
        else:
            idx_min_segment = np.argpartition(distance_measure_list, amount_to_take)[
                :amount_to_take]
            return idx_min_segment
    except:
        return np.array(range(len(distance_measure_list)))


def get_n_idx_of_min_segment(n, distance_measure_list):
    amount_to_take = n
    try:
        if amount_to_take >= len(distance_measure_list):
            idx_min_segment = np.array(range(len(distance_measure_list)))
        else:
            idx_min_segment = np.argpartition(distance_measure_list, amount_to_take)[
                :amount_to_take]
    except:
        idx_min_segment = np.array(range(len(distance_measure_list)))

    # idx_min_segment = np.argpartition(distance_measure_list, amount_to_take)[:amount_to_take]

    return idx_min_segment


def get_ds_targets(dataset):
    # Get target array from any dataset (custom, concat or subset)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    else:
        targets = []
        for dp in dataset:
            targets.append(dp[1])
        targets = np.array(targets)

    return targets


def get_unique_labels(dataset):
    # get uniqe labels from any dataset
    # if type custom dataset acessed via .classes attribute
    targets = get_ds_targets(dataset)
    unique = np.unique(targets)

    return unique


def get_label_dist(dataset):
    # get distribution of labels in dataset
    targets = get_ds_targets(dataset)
    unique, counts = np.unique(targets, return_counts=True)
    target_dist = dict(zip(unique.astype('str'), counts))

    return target_dist


def get_unique_counts(ls):
    # get distribution of elements in any list
    unique, counts = np.unique(ls, return_counts=True)
    dist = dict(zip(unique, counts))

    return dist


def get_label_idx_map(dataset):
    # Get map what indices of the dataset belong to what label
    ds_targets = get_ds_targets(dataset)
    u_labels = get_unique_labels(dataset)
    label_idx_map = {}
    for l in u_labels:
        idx = np.where(ds_targets == l)[0]
        label_idx_map[str(l)] = idx

    return label_idx_map


def get_norm_subset_idx(dataset, subset_size, distance_measure_list, segment='max'):
    """Takes a dataset and samples all classes equally according to distance measure list for the dataset"""
    assert segment == 'max' or segment == 'min', "Segment must be str 'max' or 'min'"
    distance_measure_list = distance_measure_list if isinstance(
        distance_measure_list, np.ndarray) else np.array(distance_measure_list)
    # Get label:idx map
    label_idx_map = get_label_idx_map(dataset)
    # Get label distribution
    label_dist = get_label_dist(dataset)
    # Get unique labels
    u_labels = get_unique_labels(dataset).astype('str')

    # Get how many labels you can can take max from each class for given subset size
    # Distribute the labels needed equally along the classes (as equally as possible)
    subset_amount = int(subset_size * len(dataset))
    n_class_samples_map = {label: 0 for label in u_labels}
    while subset_amount > 0:
        for l in u_labels:
            if label_dist[l] > n_class_samples_map[l] and subset_amount > 0:
                n_class_samples_map[l] = n_class_samples_map[l] + 1
                subset_amount = subset_amount - 1

    # Plot theoretical Distribution
    # x = [str(k) for k, _ in n_class_samples_map.items()]
    # y = [int(v) for _, v in n_class_samples_map.items()]
    # plt.bar(x, y)
    # plt.show()

    sample_idx = []
    for l in u_labels:
        # Get distance measure values for current label
        l_dist = distance_measure_list[label_idx_map[l]]
        # Get highest or lowest segment indices for current label
        # Use the amount for each class which was calculated in n_class_samples_map
        if segment == 'max':
            l_segment_idx = get_n_idx_of_max_segment(
                n_class_samples_map[l], l_dist)
        elif segment == 'min':
            l_segment_idx = get_n_idx_of_min_segment(
                n_class_samples_map[l], l_dist)
        # Use sample indices at label to get dataset indices from label map
        l_sample_idx = label_idx_map[l][l_segment_idx]
        # Concat into one idx list
        sample_idx = np.concatenate(
            [sample_idx, l_sample_idx], axis=None).astype(int)

    return sample_idx


def get_random_idx(dataset, n_samples, seed=10):
    # For Testing Purposes, take random n images from target for comparison
    n = n_samples
    if len(dataset) < n:
        n = len(dataset)
    idx = list(range(len(dataset)))  # Set pseudo idx list

    currRandom = Random(seed)

    r_idx = currRandom.sample(idx, n)  # Sample idx

    return r_idx


def change_ds_labels(ds, trgt):
    if hasattr(ds, 'targets'):
        ds.targets = torch.tensor([trgt for _ in range(len(ds.targets))])
    if hasattr(ds, 'labels'):
        ds.labels = torch.tensor([trgt for _ in range(len(ds.labels))])

    elif isinstance(ds, torch.utils.data.dataset.ConcatDataset):
        for i, sub_ds in enumerate(ds.datasets):
            change_ds_labels(sub_ds, trgt)

    return ds


def change_ds_transform(ds, transform):
    if hasattr(ds, 'transform'):
        assert isinstance(
            transform, torchvision.transforms.transforms.Compose), "Transform is not of type 'torchvision.transforms.transforms.Compose'"
        ds.transform = transform

    elif isinstance(ds, torch.utils.data.dataset.ConcatDataset):
        for i, sub_ds in enumerate(ds.datasets):
            change_ds_transform(sub_ds, transform)

    elif isinstance(ds, torch.utils.data.dataset.Subset) or isinstance(ds, Digit5Subset):
        # for i, sub_ds in enumerate(ds.dataset):
        change_ds_transform(ds.dataset, transform)

    return ds
