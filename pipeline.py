from random import random
from traceback import print_tb
# from unittest import result
import opts
from utils.utils import create_directory, timer, save_as_json
from utils.load_img import load_images, combine_datasets, get_dataset_origins
from sampler.random_sampler import RandomSampler
from sampler.ssim_sampler import SSIMSampler
# from utils.feature_sampler import FeatureDistSampler
# from utils.autoencoder_sampler import AutoencoderSampler
from modeling.modeling import Modeling
from utils.log import Log

# from torch.utils.data import Dataset, Subset
import sys
import os
import logging
import datetime as dt
import numpy as np
import time
import glob
import json


# # EXPERIMENT_ID = "220216-1030_Digit5_COS-normal_lenet_gpu6"
# # EXPERIMENT_ID = "220215-1450_DomainNet_EUC_resnet_gpu5"
# # EXPERIMENT_ID = "220202-1200_Digit5_AE_lenet_gpu3"
# EXPERIMENT_ID = "Trial"
# # EXPERIMENT_ID = "TESTINGRUN"

# DESCRIPTION = """Run DN normally for 2 best global methods"""

# D = "Digit5"
# # DATASETS = ["mnist", "svhn", "mnist-m", "usps", "syn"] # Digit 5
# DATASETS = ["mt", "sv"]  # Digit 5 <-- For Testing Purposes

# # D = "DomainNet"
# # # DATASETS = ['clipart', 'painting']
# # # DATASETS = ['painting', 'real']
# # # DATASETS = ['clipart', 'painting', 'real', 'sketch']
# # DATASETS = ['sketch', 'clipart', 'painting', 'real']

# # SUBSET_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# SUBSET_SIZES = [0.1]  # [0.3]

# CLASS_NORMAL = False

# TRAINING_MODEL = 'lenet'
# # TRAINING_MODEL = 'resnet18'
# # TRAINING_MODEL = 'resnet50'

# # True # False
# RANDOM_MODELING = True
# SSIM_MODELING = False
# FEATURE_COMPARISON = False
# AE_COMPARISON = False

# N_TRAINING_RUNS = 5
# N_RANDOM_SUBSAMPLES = 12
# FEATURE_CALC = ['cos', 'euclid']
# # FEATURE_CALC = ['cos']
# # FEATURE_CALC = ['euclid']
# # DIST_MODE = ['dist_avg', 'dist_min', 'dist_k_min']
# DIST_MODE = ['dist_avg']

# # VISIBLE_GPU = '4,5,6'
# VISIBLE_GPU = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU

# EXPERIMENT_ID = f"220225-1120_{D}_FTR-Real-COS-EUC-avg-inclass_{TRAINING_MODEL}_gpu{VISIBLE_GPU}"
# # EXPERIMENT_ID = "Trial"

# # ARGS
# EXPERIMENT_ARGS = {
#     "EXPERIMENT_ID": EXPERIMENT_ID,
#     "DATASETS": DATASETS,
#     "SUBSET_SIZES": SUBSET_SIZES,
#     'TRAINING_MODEL': TRAINING_MODEL,
#     "CLASS_NORMAL": CLASS_NORMAL,
#     "VISIBLE_GPU": VISIBLE_GPU,
#     "METHODS": {
#         "N_TRAINING_RUNS": N_TRAINING_RUNS,
#         "N_RANDOM_SUBSAMPLES": N_RANDOM_SUBSAMPLES,
#         "RANDOM_MODELING": RANDOM_MODELING,
#         "SSIM_MODELING": SSIM_MODELING,
#         "FEATURE_COMPARISON": FEATURE_COMPARISON,
#         "FEATURE_CALC": FEATURE_CALC,
#         "AE_COMPARISON": AE_COMPARISON,
#     }
# }

# ##### Logging #####
# # log = Log(EXPERIMENT_ID).log

# ##### Results #####
# dataset_results = {}
# dataset_results["info"] = EXPERIMENT_ARGS


class S2TFiltering():
    def __init__(self, opts):
        self.opts = opts
        self.experiment_id = opts.exp_id
        self.visible_gpu = opts.gpu
        assert opts.data in ["Digit-5", "DomainNet"], "Data argument must be 'Digit-5' or 'DomainNet'"
        # self.datasets = ["mnist", "svhn", "mnist-m", "usps", "syn"] if opts.data == "Digit-5" else ['clipart' 'painting', 'real', 'sketch']
        self.datasets = ["mnist", "svhn"] if opts.data == "Digit-5" else ['clipart' 'painting', 'real', 'sketch']
        self.training_model = 'lenet' if opts.data == "Digit-5" else 'resnet18'  # 'resnet50'
        self.opts.model = self.training_model
        self.n_training_runs = opts.n_training_runs
        # self.subset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.subset_sizes = [0.1]
        self.in_class_ranking = opts.in_class_ranking
        # Random Opts
        self.random = opts.random
        self.n_random_subsets = opts.n_random_subsets
        # SSIM Opts
        self.ssim = opts.ssim

        self.results_filepath = ''
        self.dataset_results = {}
        self.dataset_results["info"] = vars(opts)
        self.run_id = 0

        ##### Logging #####
        self.log = Log(self.experiment_id).log

    def run_s2t_filtering(self):
        # Log experiment info
        tstart = time.time()
        self.log.info(f"GPU {self.visible_gpu}")
        self.log.info(f"EXPERIMENT INFO: {vars(self.opts)}")

        # Create results directory
        results_dir = os.getcwd() + f'/results/{self.experiment_id}'
        create_directory(results_dir)
        self.results_filepath = f"{results_dir}/exp_results_{self.experiment_id}.json"

        # Load self.datasets
        datasets = {ds: load_images(
            [ds], data_trans=self.training_model, xy_combined=False) for ds in self.datasets}

        # Run for different source & target combinations
        for i_ds_run, test_dataset_id in enumerate(self.datasets):
            self.run_id = i_ds_run

            # TARGET DATA
            # Take testset if source has one, otherwise take the trainset
            test_dataset = datasets.get(test_dataset_id)[1] if datasets.get(test_dataset_id)[1] else datasets.get(test_dataset_id)[0]

            # SOURCE DATA
            train_dataset_ids = [ds for ds in self.datasets if ds != test_dataset_id]
            train_dataset_name = '_'.join(train_dataset_ids)
            train_datasets_list = [datasets.get(ds_id)[0] for ds_id in train_dataset_ids]
            train_dataset = combine_datasets(train_datasets_list)
            # Get source origins
            train_dataset_origins = get_dataset_origins(train_datasets_list)

            self.log.info(f"# START RUN - Train: {train_dataset_ids} (={len(train_dataset)}) Test: {test_dataset_id} (={len(test_dataset)}) #")
            self.dataset_results[str(self.run_id)] = {'train': train_dataset_name, 'test': test_dataset_id}

            # Run random
            if self.random:
                self.run_random_sampling(train_dataset, train_dataset_origins, test_dataset)

            if self.ssim:
                self.run_ssim_sampling(train_dataset, train_dataset_origins, test_dataset)

    def run_random_sampling(self, train_dataset, train_dataset_origins, test_dataset):
        self.log.info(f'=========Running Random========')

        # Random Data Sampling
        rs = RandomSampler(log_id=self.experiment_id)
        # Init Dict for Accuracies
        random_accuracies = {}
        # Run for all subset sizes
        for subset_size in self.subset_sizes:
            sss_str = f"{subset_size:.2f}"
            self.log.info(f'============{sss_str}===========')
            self.log.info(f"Sampling {self.n_random_subsets}x for {sss_str}")

            sss_accuracies = []
            t0 = time.time()
            for n in range(self.n_random_subsets):
                self.log.info(
                    f'====Random Subsample {n+1}/{self.n_random_subsets}====')
                # Get random subset
                train_subset = rs.get_random_subset(
                    train_dataset, subset_size, ds_origins=train_dataset_origins)
                # Model the subset
                m = Modeling(self.visible_gpu, log_id=self.experiment_id)
                acc = m.run_modeling(
                    train_subset, test_dataset, self.training_model)
                # Append all accs to current list
                sss_accuracies.append(acc)

            # Append subset size acc list to random_accuracies dict
            random_accuracies[sss_str] = sss_accuracies
            t1 = time.time()
            self.log.info(f'### Total Time for Training {self.n_random_subsets}x for {sss_str}%: {timer(t0,t1)} ###')
            # Append dict of accs for different sss to dataset_restults
            self.dataset_results[str(self.run_id)]["random"] = {"n_subsamples": self.n_random_subsets, "acc": random_accuracies}
            # Save intermediate results
            save_as_json(data=self.dataset_results, filepath=self.results_filepath)

        for k, v in random_accuracies.items():
            print(f"{k} : {v}")
            self.log.info(f"{k} : {v}")

    def run_ssim_sampling(self, train_dataset, train_dataset_origins, test_dataset):
        self.log.info(f'=========Running SSIM=========')

        # SSIM Data Sampling
        s = SSIMSampler(train_dataset, test_dataset, log_id=self.experiment_id)

        for ssim_mode in ['ssim_avg', 'ssim_max', 'ssim_k_max']:
            self.log.info(f'====== Mode: {ssim_mode} =====')
            # Init Dict for Accuracies
            ssim_accuracies = {}
            # Run for all subset sizes
            for subset_size in self.subset_sizes:
                sss_str = f"{subset_size:.2f}"
                self.log.info(f'============{sss_str}===========')
                # Take subset from ranking according to mode
                train_subset = s.get_max_ssim_subset(train_dataset, subset_size, mode_id=ssim_mode, class_equal=self.in_class_ranking, ds_origins=train_dataset_origins)

                sss_accuracies = []
                t0 = time.time()
                # run it n_training_runs for non deterministics
                for _ in range(self.n_training_runs):
                    # Model the subset
                    m = Modeling(self.visible_gpu, log_id=self.experiment_id)
                    acc = m.run_modeling(train_subset, test_dataset, self.training_model)
                    # Append all accs to current list
                    sss_accuracies.append(acc)

                # Append list to ssim_accuracies dict
                ssim_accuracies[sss_str] = sss_accuracies
                t1 = time.time()
                self.log.info(f'### Total Time for Training {self.n_training_runs}x for {sss_str}%: {timer(t0,t1)} ###')
                # Append dict of accs for different sss to dataset_restults
                self.dataset_results[str(self.run_id)][ssim_mode] = {"n_training_runs": self.n_training_runs, "acc": ssim_accuracies}
                # Save intermediate results in case training breaks
                save_as_json(data=self.dataset_results, filepath=self.results_filepath)

            for k, v in ssim_accuracies.items():
                print(f"{k} : {v}")
                self.log.info(f"{k} : {v}")


# def old_run_s2t_filtering(opt):
#     self.experiment_id = opt.exp_id
#     VISIBLE_GPU = opt.gpu
#     assert opt.data in [
#         "Digit-5", "DomainNet"], "data argument must be 'Digit-5' or 'DomainNet'"
#     self.datasets = ["mnist", "svhn", "mnist-m", "usps",
#                      "syn"] if opt.data == "Digit-5" else ['clipart', 'painting', 'real', 'sketch']

#     log = Log(self.experiment_id).log

#     # Log experiment info
#     tstart = time.time()
#     log.info(f"GPU {VISIBLE_GPU}")
#     log.info(f"EXPERIMENT INFO: {EXPERIMENT_ARGS}")

#     # Create results directory
#     results_dir = os.getcwd() + f'/results/{self.experiment_id}'
#     create_directory(results_dir)
#     filepath = f"{results_dir}/exp_results_{self.experiment_id}.json"

#     # Load datasets
#     datasets = {ds: load_images(
#         [ds], data_trans=self.training_model, xy_combined=False) for ds in self.datasets}

#     # Run for different source & target combinations
#     for i_ds, test_dataset_id in enumerate(self.datasets):

#         train_dataset_ids = [
#             ds for ds in self.datasets if ds != test_dataset_id]
#         train_dataset_name = '_'.join(train_dataset_ids)

#         # TRAIN DATA
#         # Choose Data for Training the models for acc comparison
#         train_datasets_list = [datasets.get(
#             ds_id)[0] for ds_id in train_dataset_ids]
#         train_dataset = combine_datasets(train_datasets_list)
#         # Get source origins
#         train_dataset_origins = get_dataset_origins(train_datasets_list)

#         # TEST DATA
#         # Take testset if it exists, otherwise, take the trainset
#         test_dataset = datasets.get(test_dataset_id)[1] if datasets.get(
#             test_dataset_id)[1] else datasets.get(test_dataset_id)[0]

#         log.info(
#             f"# START RUN - Train: {train_dataset_ids} (={len(train_dataset)}) Test: {test_dataset_id} (={len(test_dataset)}) #")
#         dataset_results[str(i_ds)] = {
#             'train': train_dataset_name, 'test': test_dataset_id}

#     #     # Subsets for Testing Purposes
#     #     # train_dataset = Subset(train_dataset, list(range(100)))
#     #     # test_dataset = Subset(test_dataset, list(range(1000)))

#         ##################
#         ##### Random #####
#         ##################
#         if RANDOM_MODELING:
#             pass
#             # log.info(f'=========Running Random========')
#             # # Init Dict for Accuracies
#             # random_accuracies = {}
#             # # Random Data Sampling
#             # rs = RandomSampler(log_id=self.experiment_id)
#             # n_subsamples = N_RANDOM_SUBSAMPLES
#             # for subset_size in self.subset_sizes:
#             #     sss_str = f"{subset_size:.2f}"
#             #     log.info(f'============{sss_str}===========')
#             #     log.info(f"Sampling {n_subsamples}x for {sss_str}")
#             #     # Get list of n random subsets
#             #     #random_subset_ds_list = [rs.get_random_subset(train_dataset, subset_size, ds_origins=train_dataset_origins) for _ in range(n_subsamples)]

#             #     sss_accuracies = []
#             #     t0 = time.time()
#             #     # for train_subset in random_subset_ds_list:
#             #     for n in range(n_subsamples):
#             #         log.info(f'====Random Subsample {n+1}/{n_subsamples}====')
#             #         # Get list of n random subsets
#             #         train_subset = rs.get_random_subset(
#             #             train_dataset, subset_size, ds_origins=train_dataset_origins)
#             #         m = Modeling(VISIBLE_GPU, log_id=self.experiment_id)
#             #         acc = m.run_modeling(
#             #             train_subset, test_dataset, self.training_model)
#             #         # Append all accs to current list
#             #         sss_accuracies.append(acc)

#             #     # Append list to random_accuracies dict
#             #     random_accuracies[sss_str] = sss_accuracies
#             #     t1 = time.time()
#             #     log.info(
#             #         f'### Total Time for Training {n_subsamples}x for {sss_str}%: {timer(t0,t1)} ###')
#             #     # Append dict of accs for different sss to dataset_restults
#             #     dataset_results[str(i_ds)]["random"] = {
#             #         "n_subsamples": n_subsamples, "acc": random_accuracies}
#             #     # Save intermediate results in case training breaks
#             #     save_as_json(data=dataset_results, filepath=filepath)
#             # for k, v in random_accuracies.items():
#             #     print(f"{k} : {v}")
#             #     log.info(f"{k} : {v}")

#     #     ##################
#     #     ###### SSIM ######
#     #     ##################
#     #     if SSIM_MODELING:
#     #         log.info(f'=========Running SSIM=========')
#     #         s = SSIMSampler(train_dataset, test_dataset, log_id=self.experiment_id)

#     #         for ssim_mode in ['ssim_avg', 'ssim_max', 'ssim_k_max']:
#     #             log.info(f'====== Mode: {ssim_mode} =====')
#     #             ssim_accuracies = {}
#     #             for subset_size in self.subset_sizes:
#     #                 sss_str = f"{subset_size:.2f}"
#     #                 log.info(f'============{sss_str}===========')
#     #                 train_subset = s.get_max_ssim_subset(
#     #                     train_dataset, subset_size, mode_id=ssim_mode, class_equal=CLASS_NORMAL, ds_origins=train_dataset_origins)

#     #                 # To see if there are any non deterministics when modeling run it n_training_runs
#     #                 n_training_runs = N_TRAINING_RUNS
#     #                 sss_accuracies = []
#     #                 t0 = time.time()
#     #                 for _ in range(n_training_runs):
#     #                     # Run Modeling
#     #                     m = Modeling(VISIBLE_GPU, log_id=self.experiment_id)
#     #                     acc = m.run_modeling(
#     #                         train_subset, test_dataset, self.training_model)
#     #                     # Append all accs to current list
#     #                     sss_accuracies.append(acc)

#     #                 # Append list to ssim_accuracies dict
#     #                 ssim_accuracies[sss_str] = sss_accuracies
#     #                 t1 = time.time()
#     #                 log.info(
#     #                     f'### Total Time for Training {n_training_runs}x for {sss_str}%: {timer(t0,t1)} ###')
#     #                 # Append dict of accs for different sss to dataset_restults
#     #                 dataset_results[str(i_ds)][ssim_mode] = {
#     #                     "n_training_runs": n_training_runs, "acc": ssim_accuracies}
#     #                 # Save intermediate results in case training breaks
#     #                 save_as_json(data=dataset_results, filepath=filepath)
#     #             for k, v in ssim_accuracies.items():
#     #                 print(f"{k} : {v}")
#     #                 log.info(f"{k} : {v}")

#     #     ######################
#     #     ## Feature Distance ##
#     #     ######################
#     #     if FEATURE_COMPARISON and loop == 0:
#     #         loop += 1
#     #         log.info(f'=========Running Feature Distance=========')

#     #         print("Running Feat...")
#     #         for calc in FEATURE_CALC:
#     #             log.info(f'====== Calculation: {calc} =====')

#     #             fds = FeatureDistSampler(
#     #                 train_dataset, test_dataset, calc, VISIBLE_GPU, log_id=self.experiment_id)

#     #             # for dist_mode in ['dist_avg', 'dist_min', 'dist_k_min']:
#     #             for dist_mode in DIST_MODE:  # TODO: <====== HIER IMMER EINEN VERWENDEN
#     #                 print(f"Running do {calc}: {dist_mode}")
#     #                 log.info(f'====== Mode: {dist_mode} =====')
#     #                 fdist_accuracies = {}
#     #                 for subset_size in self.subset_sizes:
#     #                     sss_str = f"{subset_size:.2f}"
#     #                     log.info(f'============{sss_str}===========')

#     #                     train_subset = fds.get_min_dist_subset(
#     #                         train_dataset, subset_size, mode_id=dist_mode, class_equal=CLASS_NORMAL, ds_origins=train_dataset_origins)

#     #                     # To see if there are any non deterministics when modeling run it n_training_runs
#     #                     n_training_runs = N_TRAINING_RUNS
#     #                     sss_accuracies = []
#     #                     t0 = time.time()
#     #                     for _ in range(n_training_runs):

#     #                         m = Modeling(VISIBLE_GPU, log_id=self.experiment_id)
#     #                         acc = m.run_modeling(
#     #                             train_subset, test_dataset, self.training_model)
#     #                         # Append all accs to current list
#     #                         sss_accuracies.append(acc)

#     #                     # Append list to ssim_accuracies dict
#     #                     fdist_accuracies[sss_str] = sss_accuracies
#     #                     t1 = time.time()
#     #                     log.info(
#     #                         f'### Total Time for Training {n_training_runs}x for {sss_str}%: {timer(t0,t1)} ###')
#     #                     # Append dict of accs for different sss to dataset_restults
#     #                     dataset_results[str(i_ds)][f"{dist_mode}_{calc}"] = {
#     #                         "n_training_runs": n_training_runs, "acc": fdist_accuracies}
#     #                     # Save intermediate results in case training breaks
#     #                     save_as_json(data=dataset_results, filepath=filepath)
#     #                 for k, v in fdist_accuracies.items():
#     #                     print(f"{k} : {v}")
#     #                     log.info(f"{k} : {v}")

#     #     #################
#     #     ## Autoencoder ##
#     #     #################
#     #     if AE_COMPARISON:
#     #         log.info(f'=======Running Autoencoder=======')

#     #         target_img_channels = test_dataset[0][0].shape[0]
#     #         ae = AutoencoderSampler(
#     #             train_dataset, test_dataset, target_img_channels, VISIBLE_GPU, log_id=self.experiment_id)

#     #         ae_accuracies = {}
#     #         for subset_size in self.subset_sizes:
#     #             sss_str = f"{subset_size:.2f}"
#     #             log.info(f'============{sss_str}===========')
#     #             train_subset = ae.get_min_autoencoder_subset(
#     #                 train_dataset, subset_size, class_equal=CLASS_NORMAL, ds_origins=train_dataset_origins)

#     #             # To see if there are any non deterministics when modeling run it n_training_runs
#     #             n_training_runs = N_TRAINING_RUNS
#     #             sss_accuracies = []
#     #             t0 = time.time()
#     #             for _ in range(n_training_runs):
#     #                 # Run Modeling
#     #                 m = Modeling(VISIBLE_GPU, log_id=self.experiment_id)
#     #                 acc = m.run_modeling(
#     #                     train_subset, test_dataset, self.training_model)
#     #                 # Append all accs to current list
#     #                 sss_accuracies.append(acc)

#     #             # Append list to ae_accuracies dict
#     #             ae_accuracies[sss_str] = sss_accuracies
#     #             t1 = time.time()
#     #             log.info(
#     #                 f'### Total Time for Training {n_training_runs}x for {sss_str}%: {timer(t0,t1)} ###')
#     #             # Append dict of accs for different sss to dataset_restults
#     #             dataset_results[str(i_ds)]["autoencoder"] = {
#     #                 "n_training_runs": n_training_runs, "acc": ae_accuracies}
#     #             # Save intermediate results in case training breaks
#     #             save_as_json(data=dataset_results, filepath=filepath)

#     #         for k, v in ae_accuracies.items():
#     #             print(f"{k} : {v}")
#     #             log.info(f"{k} : {v}")

#     # tend = time.time()
#     # print("+ + + + + + + + + + + + + + + +")
#     # print(f"Total Experiment time: {timer(tstart,tend)}")
#     # log.info(f"+++ Total Experiment time: {timer(tstart,tend)} +++")
#     # print(dataset_results)
#     # # Save as results
#     # save_as_json(data=dataset_results, filepath=filepath)
#     # print("+ + + + + + + + + + + + + + + +")

#     print("Done")
#     print()
#     print(vars(opt))

# if __name__ == "__main__":
#     log.info("Starting run_experiments.py...")
#     print("Starting...")
#     run_s2t_filtering()
