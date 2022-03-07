from utils.datasets import MNISTDataset, SVHNDataset, USPSDataset, SYNDataset, MNISTMDataset, Digit5Subset
from utils.datasets import ClipartDataset, PaintingDataset, RealDataset, SketchDataset
from utils.utils import create_directory
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np



data_transforms = {
    "lenet": transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ]),
    "resnet": transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
}


def load_images(dataset_ids_list: list, data_trans="resnet", xy_combined=False, subset=True):
    """Load image datasets.

    Loads images datasets according to ids given. Can be combinations of 
    ["mnist", "svhn", "mnist-m", "usps", "syn"] or ['clipart' 'painting', 
    'real', 'sketch']. Transforms the data to comply with resnet- or lenet-
    network archtecure. 

    Args:
        dataset_ids_list (list): ids of what datasets to load
        data_trans (torchvision.transforms): how the data is transformed when
            loaded (default is "resnet") 
        xy_combined (bool): whether train and test data should be combined to 
            one dataset if possible (default is False)
        subset (bool): whether Digit-5 subsets should be taken according to
            literature benchmarks

    Returns:
        touple: touple of train and test datasets
    """

    assert isinstance(dataset_ids_list, list), 'dataset_ids_list is not type list'
    assert data_trans in ["lenet", "resnet", "resnet18", "resnet50"], 'Data Transform should be one of "lenet", "resnet", "resnet18", "resnet50"'
    create_directory('./data/digit-5')

    transform_images = 'resnet' if data_trans in ["resnet18", "resnet50"] else data_trans

    train_data = []
    test_data = []
    for dataset in dataset_ids_list:
        ### DIGIT 5 ###
        if dataset == "mnist" or dataset == "mt":
            train = MNISTDataset(
                split='train', transform=data_transforms.get(transform_images, None))
            test = MNISTDataset(
                split='test', transform=data_transforms.get(transform_images, None))
        if dataset == "svhn" or dataset == "sv":
            train = SVHNDataset(
                split='train', transform=data_transforms.get(transform_images, None))
            test = SVHNDataset(
                split='test', transform=data_transforms.get(transform_images, None))
        if dataset == "usps" or dataset == "up":
            train = USPSDataset(
                split="train", transform=data_transforms.get(transform_images, None))
            test = USPSDataset(
                split="test", transform=data_transforms.get(transform_images, None))
        if dataset == "syn" or dataset == "sy":
            train = SYNDataset(
                split="train", transform=data_transforms.get(transform_images, None))
            test = SYNDataset(
                split="test", transform=data_transforms.get(transform_images, None))
        if dataset == "mnist-m" or dataset == "mm":
            train = MNISTMDataset(
                split="train", transform=data_transforms.get(transform_images, None))
            test = MNISTMDataset(
                split="test", transform=data_transforms.get(transform_images, None))

        # Take subset according to Peng et al. "Moment Matching" source code
        if subset and dataset in ["mnist", "svhn", "mnist-m", "syn", "mt", "sv", "mm", "sy", "mt-twin"]:
            train = Digit5Subset(train, list(range(25000)))
            test = Digit5Subset(test, list(range(9000)))
        if subset and dataset in ["usps", "up"]:
            train = combine_datasets([train, test])
            train.name = "usps"
            test = None

        ### DOMAIN NET ###
        if dataset == "clipart" or dataset == "c":
            train = ClipartDataset(
                transform=data_transforms.get(transform_images, None))
            test = None
        if dataset == "painting" or dataset == "p":
            train = PaintingDataset(
                transform=data_transforms.get(transform_images, None))
            test = None
        if dataset == "real" or dataset == "r":
            train = RealDataset(
                transform=data_transforms.get(transform_images, None))
            test = None
        if dataset == "sketch" or dataset == "s":
            train = SketchDataset(
                transform=data_transforms.get(transform_images, None))
            test = None

        # Info on loaded data
        n_test = len(test) if test else 0
        print(f"Loaded {dataset} ({transform_images}): train {len(train)}; test {n_test}")

        # Combine x and y data
        if xy_combined:
            train = combine_datasets([train, test])
            test = None

        # Append train datasets/test datasets together
        train_data.append(train)
        if test:
            test_data.append(test)

    # Concat training and test data
    train_dataset = combine_datasets(train_data)
    test_dataset = combine_datasets(test_data) if test_data else None

    return train_dataset, test_dataset


def combine_datasets(datasets: list):
    """Combine list of datasets to one dataset.

    Args:
        datasets (list): list of datasets

    Returns:
        torch.utils.data.Dataset: dataset
    """
    if len(datasets) == 1:
        return datasets[0]
    else:
        datasets = [ds for ds in datasets if ds is not None]
        combined_dataset = ConcatDataset(datasets)

        return combined_dataset


def get_dataset_origins(datasets):
    """Generate list of source origins for list of datasets.

    Generated for logging reasons and traceability of generated 
    subsets.

    Args:
        datasets (list): list of datasets

    Returns:
        numpy.ndarray): array of dataset names with length of combined datasets
    """
    datasets = [ds for ds in datasets if ds is not None]

    origins_list = []
    for ds in datasets:
        try:
            name = ds.name
            origins_list = origins_list + [name for _ in range(len(ds))]
        except:
            raise Exception("Could not append dataset origin.")

    origins_list = np.array(origins_list)

    return origins_list
