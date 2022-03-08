from utils.utils import create_directory, timer, save_as_json
from utils.load_img import load_images, combine_datasets, get_dataset_origins
from sampler.random_sampler import RandomSampler
from sampler.ssim_sampler import SSIMSampler
from sampler.feature_sampler import FeatureDistSampler
from sampler.autoencoder_sampler import AutoencoderSampler
from modeling.modeling import Modeling
from utils.log import Log
import os
import numpy as np
import time


class S2TFiltering():
    """Source to target filtering pipleine class to steer experimentation"""

    def __init__(self, opts):
        self.opts = opts
        self.experiment_id = opts.exp_id
        self.visible_gpu = opts.gpu
        assert opts.data in ["Digit-5", "DomainNet"], "Data argument must be 'Digit-5' or 'DomainNet'"
        self.datasets = ["mnist", "svhn", "mnist-m", "usps", "syn"] if opts.data == "Digit-5" else ['clipart' 'painting', 'real', 'sketch']
        self.training_model = 'lenet' if opts.data == "Digit-5" else 'resnet18'  # 'resnet50'
        self.opts.model = self.training_model
        self.n_training_runs = opts.n_training_runs
        self.subset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
        self.in_class_ranking = opts.in_class_ranking

        self.random = opts.random
        self.n_random_subsets = opts.n_random_subsets
        self.ssim = opts.ssim
        self.feature = opts.feature
        self.dist_measure = opts.dist_measure
        self.autoencoder = opts.autoencoder

        self.results_filepath = ''
        self.dataset_results = {}
        self.dataset_results["info"] = vars(opts)
        self.run_id = 0

        ##### Logging #####
        self.log = Log(self.experiment_id).log

    def run_s2t_filtering(self):
        """Execution of the s2t-similarity filtering pipeline.

        Runs the sampling methods set in the Arguments for the given datasets
        and all its source/target combinations, plus modeling for all
        subsets created. Function args are taken from class.

        Args:
            None

        Returns:
            dict: final resulst for each source/target combination, 
                method and subset size specified
        """

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
            # Run SSIM
            if self.ssim:
                self.run_ssim_sampling(train_dataset, train_dataset_origins, test_dataset)
            # Run Feature Dist
            if self.feature:
                self.run_feature_dist_sampling(train_dataset, train_dataset_origins, test_dataset)
            # Run Autoencoder
            if self.autoencoder:
                self.run_autoencoder_sampling(train_dataset, train_dataset_origins, test_dataset)

        tend = time.time()
        self.log.info(f"+++ Total Experiment time: {timer(tstart,tend)} +++")
        # Save final results
        save_as_json(data=self.dataset_results, filepath=self.results_filepath)

        return self.dataset_results

    def run_random_sampling(self, train_dataset, train_dataset_origins, test_dataset):
        """Generates subsets from source data via random sampling.

        Subsets created according to subset sizes via random sampling. Model training
        performed with subsets and evaluated on target data. Results saved as json.

        Args:
            train_dataset (torch.utils.data.Dataset): combined source datasets
            train_dataset_origins (numpy.ndarray): array of dataset names
            test_dataset (torch.utils.data.Dataset): target dataset

        Returns:
            dict: current resulst for each source/target combination, 
                random sampling and subset size specified
        """
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
            self.log.info(f"{k} : {v}")

        return self.dataset_results

    def run_ssim_sampling(self, train_dataset, train_dataset_origins, test_dataset):
        """Generates subsets from source data via Structural Similarity sampling.

        Subsets created according to subset sizes via SSIM sampling. Model training
        performed with subsets and evaluated on target data. Results saved as json.

        Args:
            train_dataset (torch.utils.data.Dataset): combined source datasets
            train_dataset_origins (numpy.ndarray): array of dataset names
            test_dataset (torch.utils.data.Dataset): target dataset

        Returns:
            dict: current resulst for each source/target combination, 
                random sampling and subset size specified
        """
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
                self.log.info(f"{k} : {v}")

        return self.dataset_results

    def run_feature_dist_sampling(self, train_dataset, train_dataset_origins, test_dataset):
        """Generates subsets from source data via learned feature space distance sampling.

        Subsets created according to subset sizes via learned feature space distance sampling.
        Model training performed with subsets and evaluated on target data. Results saved as json.

        Args:
            train_dataset (torch.utils.data.Dataset): combined source datasets
            train_dataset_origins (numpy.ndarray): array of dataset names
            test_dataset (torch.utils.data.Dataset): target dataset

        Returns:
            dict: current resulst for each source/target combination, 
                random sampling and subset size specified
        """
        self.log.info(f'=========Running Feature Distance=========')

        # Learned Feature Space Distance Sampler
        fds = FeatureDistSampler(train_dataset, test_dataset, dist=self.dist_measure, visible_gpus=self.visible_gpu, log_id=self.experiment_id)

        for dist_mode in ['dist_avg', 'dist_min', 'dist_k_min']:
            self.log.info(f'====== Mode: {self.dist_measure} - {dist_mode} =====')
            # Init Dict for Accuracies
            fdist_accuracies = {}
            # Run for all subset sizes
            for subset_size in self.subset_sizes:
                sss_str = f"{subset_size:.2f}"
                self.log.info(f'============{sss_str}===========')
                # Take subset from ranking according to mode
                train_subset = fds.get_min_dist_subset(train_dataset, subset_size, mode_id=dist_mode, class_equal=self.in_class_ranking, ds_origins=train_dataset_origins)

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
                fdist_accuracies[sss_str] = sss_accuracies
                t1 = time.time()
                self.log.info(f'### Total Time for Training {self.n_training_runs}x for {sss_str}%: {timer(t0,t1)} ###')
                # Append dict of accs for different sss to dataset_restults
                self.dataset_results[str(self.run_id)][f"{dist_mode}_{self.dist_measure}"] = {"n_training_runs": self.n_training_runs, "acc": fdist_accuracies}
                # Save intermediate results in case training breaks
                save_as_json(data=self.dataset_results, filepath=self.results_filepath)
            for k, v in fdist_accuracies.items():
                self.log.info(f"{k} : {v}")

        return self.dataset_results

    def run_autoencoder_sampling(self, train_dataset, train_dataset_origins, test_dataset):
        """Generates subsets from source data via autoencoder loss sampling.

        Subsets created according to subset sizes via autoencoder loss sampling. 
        Model training performed with subsets and evaluated on target data. 
        Results saved as json.

        Args:
            train_dataset (torch.utils.data.Dataset): combined source datasets
            train_dataset_origins (numpy.ndarray): array of dataset names
            test_dataset (torch.utils.data.Dataset): target dataset

        Returns:
            dict: current resulst for each source/target combination, 
                random sampling and subset size specified
        """
        self.log.info(f'=======Running Autoencoder=======')

        # Set target channels for AE
        target_img_channels = test_dataset[0][0].shape[0]
        # Train target autoencoder and eval source by initializing it
        ae = AutoencoderSampler(train_dataset, test_dataset, target_img_channels, self.visible_gpu, log_id=self.experiment_id)

        # Init Dict for Accuracies
        ae_accuracies = {}
        # Run for all subset sizes
        for subset_size in self.subset_sizes:
            sss_str = f"{subset_size:.2f}"
            self.log.info(f'============{sss_str}===========')
            # Take subset from ranking
            train_subset = ae.get_min_autoencoder_subset(train_dataset, subset_size, class_equal=self.in_class_ranking, ds_origins=train_dataset_origins)

            sss_accuracies = []
            t0 = time.time()
            # run it n_training_runs for non deterministics
            for _ in range(self.n_training_runs):
                # Run Modeling
                m = Modeling(self.visible_gpu, log_id=self.experiment_id)
                acc = m.run_modeling(
                    train_subset, test_dataset, self.training_model)
                # Append all accs to current list
                sss_accuracies.append(acc)

            # Append list to ae_accuracies dict
            ae_accuracies[sss_str] = sss_accuracies
            t1 = time.time()
            self.log.info(f'### Total Time for Training {self.n_training_runs}x for {sss_str}%: {timer(t0,t1)} ###')
            # Append dict of accs for different sss to dataset_restults
            self.dataset_results[str(self.run_id)]["autoencoder"] = {"n_training_runs": self.n_training_runs, "acc": ae_accuracies}
            # Save intermediate results in case training breaks
            save_as_json(data=self.dataset_results, filepath=self.results_filepath)

        for k, v in ae_accuracies.items():
            self.log.info(f"{k} : {v}")

        return self.dataset_results
