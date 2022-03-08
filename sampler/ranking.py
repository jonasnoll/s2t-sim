from utils.utils import get_label_dist, get_label_idx_map, get_unique_labels
import numpy as np


def get_items_idx_of_max_segment(segment_size, distance_measure_list):
    """Returns a list of indices for the highest values within a list.

    This is done to realize the image ranking in combination with 
    the subset generation. Indices are used to create the 
    subset from the datasets.

    Args: 
        segment_size (float): percentage of how much to take (how large 
            the segment size should be)

    Returns: 
        numpay.ndarray: list of indices that belong to segment
    """
    amount_to_take = int(len(distance_measure_list)*segment_size)
    try:
        # Check if more is asked then available
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
    """Returns a list of indices for the highest values within a list.

    This is done to realize the image ranking in combination with 
    the subset generation. Indices are used to create the 
    subset from the datasets.

    Args: 
        n (int): amount of how much to take (how large 
            the segment size should be)
        distance_measure_list (list / numpy.ndarray): list of image similarity measures
            to perform ranking with
    Returns: 
        numpay.ndarray: list of indices that belong to segment
    """
    amount_to_take = n
    try:
        if amount_to_take >= len(distance_measure_list):
            idx_max_segment = np.array(range(len(distance_measure_list)))
        else:
            if amount_to_take == 0:
                amount_to_take = 1
            idx_max_segment = np.argpartition(distance_measure_list, -amount_to_take)[-amount_to_take:]
    except:
        idx_max_segment = np.array(range(len(distance_measure_list)))

    return idx_max_segment


def get_items_idx_of_min_segment(segment_size, distance_measure_list):
    """Returns a list of indices for the lowest values within a list.

    This is done to realize the image ranking in combination with 
    the subset generation. Indices are used to create the 
    subset from the datasets.

    Args: 
        segment_size (float): percentage of how much to take (how large 
            the segment size should be)
        distance_measure_list (list / numpy.ndarray): list of image similarity measures
            to perform ranking with

    Returns: 
        numpay.ndarray: list of indices that belong to segment
    """
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
    """Returns a list of indices for the highest values within a list.

    This is done to realize the image ranking in combination with 
    the subset generation. Indices are used to create the 
    subset from the datasets.

    Args: 
        n (int): amount of how much to take (how large 
            the segment size should be)
        distance_measure_list (list / numpy.ndarray): list of image similarity measures
            to perform ranking with
    Returns: 
        numpay.ndarray: list of indices that belong to segment
    """
    amount_to_take = n
    try:
        if amount_to_take >= len(distance_measure_list):
            idx_min_segment = np.array(range(len(distance_measure_list)))
        else:
            idx_min_segment = np.argpartition(distance_measure_list, amount_to_take)[
                :amount_to_take]
    except:
        idx_min_segment = np.array(range(len(distance_measure_list)))

    return idx_min_segment


def get_norm_subset_idx(dataset, subset_size, distance_measure_list, segment='max'):
    """Return list of indices so classes are qually distributed

    Done to realize in-class ranking method. Takes a dataset and 
    samples all classes equally according to distance measure list for the dataset.

    Args:
        dataset (asdf): dataset containing images and labels
        segment_size (float): percentage of how much to take (how large 
            the segment size should be)
        distance_measure_list (list / numpy.ndarray): list of image similarity measures
            to perform ranking with
    Returns: 
        numpay.ndarray: list of indices that belong to needed segment AND 
            are equally distributed
    """
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

    sample_idx = []
    for l in u_labels:
        # Get distance measure values for current label
        l_dist = distance_measure_list[label_idx_map[l]]
        # Get highest or lowest segment indices for current label
        # Use the amount for each class which was calculated in n_class_samples_map
        if segment == 'max':
            l_segment_idx = get_n_idx_of_max_segment(n_class_samples_map[l], l_dist)
        elif segment == 'min':
            l_segment_idx = get_n_idx_of_min_segment(n_class_samples_map[l], l_dist)
        # Use sample indices at label to get dataset indices from label map
        l_sample_idx = label_idx_map[l][l_segment_idx]
        # Concat into one idx list
        sample_idx = np.concatenate([sample_idx, l_sample_idx], axis=None).astype(int)

    return sample_idx