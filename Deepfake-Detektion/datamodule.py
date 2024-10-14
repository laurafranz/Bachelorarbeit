"""
Module for handling data loading, balancing, and sampling in PyTorch-based deep learning models.

This module provides classes for custom data loading, dataset management, and data balancing using techniques such as 
SMOTE, oversampling, undersampling, and weighted sampling. It also includes a PyTorch Lightning DataModule that integrates 
with PyTorch Lightning to manage the data pipeline for training, validation, and testing phases. 

Classes:
    - CustomDataLoader: Handles loading of training, validation, and test datasets, with feature extraction and reshaping for ROI.
    - CustomDataset: A PyTorch dataset class that loads features and labels for training.
    - DataModule: A PyTorch Lightning DataModule that creates DataLoader instances for training, validation, and testing phases.
    - ImbalancedDataSamplerFactory: Creates different samplers (balanced, weighted) for handling imbalanced datasets.
    - BalancedBatchSampler: A sampler that generates balanced batches of data ensuring class representation within each batch.
    - WeightedBatchSampler: A sampler that generates weighted batches, giving higher sampling probabilities to underrepresented classes.
    - DataBalancer: Applies balancing techniques such as SMOTE, oversampling, and undersampling to balance class distributions.
"""

import warnings
from typing import List, Optional, Tuple, Union, Any, Iterator
from pathlib import Path
import random
import logging
import multiprocessing as mp
import pytorch_lightning as pl
from imblearn.over_sampling import SMOTE
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, BatchSampler, Sampler

from rppg_facepatches.LSTM import config
from rppg_facepatches.LSTM.config import PathConfig

# Suppress specific warnings related to 'train_dataloader' and 'val_dataloader' num_workers suggestion
warnings.filterwarnings("ignore", message=".*does not have many workers which may be a bottleneck.*")


class CustomDataLoader:
    """
    A custom data loader class that handles the creation and loading of training, validation, and test datasets.

    Attributes:
        video_output_dir (Path): Directory where the video output data is stored.
        dataset_types (List[str]): Types of datasets to load, e.g., 'real' and 'deepfake'.
        train_dataset (Optional[Dataset]): The training dataset.
        val_dataset (Optional[Dataset]): The validation dataset.
        test_dataset (Optional[Dataset]): The test dataset.
    """

    def __init__(self):
        self.video_output_dir: Path = PathConfig.VIDEO_OUTPUT_DIR
        self.dataset_types = ['real', 'deepfake']
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
    
    def _create_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Internal method to create the training, validation, and test datasets.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: A tuple containing the training, validation, and test datasets.
        """

        train_features, train_labels = self.load_train_data()
        val_features, val_labels = self.load_val_data()
        test_features, test_labels = self.load_test_data()

        return (CustomDataset(train_features, train_labels), 
                CustomDataset(val_features, val_labels), 
                CustomDataset(test_features, test_labels))
    
    def create_datasets(self, stage: Optional[str] = None) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Public method to create and return the datasets based on the current stage.
        Datasets are only created if they haven't been created yet.

        Args:
            stage (Optional[str]): The current stage ('fit', 'validate', 'test'). Default is None.

        Returns:
            Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]: 
            A tuple containing the training, validation, and test datasets based on the stage.
        """

        # if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
        #     self.train_dataset, self.val_dataset, self.test_dataset = self._create_datasets()
        # return self.train_dataset, self.val_dataset, self.test_dataset
        # Create datasets based on the current stage
        if stage == 'fit' or stage is None:
            if self.train_dataset is None or self.val_dataset is None:
                self.train_dataset, self.val_dataset = self._create_train_val_datasets()

        if stage == 'test' or stage is None:
            if self.test_dataset is None:
                self.test_dataset = self._create_test_dataset()

        # Return datasets based on the requested stage
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def _create_train_val_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Internal method to create the training and validation datasets.

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
        """
        train_features, train_labels = self.load_train_data()
        val_features, val_labels = self.load_val_data()

        return (CustomDataset(train_features, train_labels), 
                CustomDataset(val_features, val_labels))

    def _create_test_dataset(self) -> Dataset:
        """
        Internal method to create the test dataset.

        Returns:
            Dataset: The test dataset.
        """
        test_features, test_labels = self.load_test_data()
        return CustomDataset(test_features, test_labels)

    def load_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and returns the training data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the features and labels for the training dataset.
        """

        return self._load_data('train')

    def load_val_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and returns the validation data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the features and labels for the validation dataset.
        """

        return self._load_data('val')

    def load_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and returns the test data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the features and labels for the test dataset.
        """

        return self._load_data('test')
    
    def _load_data(self, video_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal method to load data based on the dataset type.

        Args:
            dataset_type (str): The type of dataset to load ('train', 'val', 'test').

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the features and labels for the specified dataset.
        """

        features: List[np.ndarray] = []
        labels: List[np.longlong] = []

        for dataset_type in self.dataset_types:
            data = self._extract_features(dataset_type, video_type)
            data = self._reshape_data(data)

            # Assign labels based on the category (real:0, deepfake:1)
            dataset_labels = np.zeros(len(data), dtype=np.longlong) if dataset_type == "real" else np.ones(len(data), dtype=np.longlong)

            if data.size > 0:
                features.append(data)
                labels.append(dataset_labels)

        # Concatenate all features and labels across categories
        if features:
            features = np.concatenate(features, axis=0)
        if labels:
            labels = np.concatenate(labels, axis=0)

        # Convert to torch tensors, ensuring features are of type torch.float32
        features_tensor = torch.from_numpy(features).float()  # Explicitly convert to torch.float32
        labels_tensor = torch.from_numpy(labels).long()  # Convert labels to torch.int64 (long in PyTorch)

        # return features, labels # torch.tensor(features), torch.tensor(labels)
        return features_tensor, labels_tensor

    def _reshape_data(self, data: List[np.ndarray]) -> np.ndarray:
        """
        Reshapes the input data based on the configuration.

        Args:
            data (List[np.ndarray]): A list of numpy arrays representing feature data.

        Returns:
            np.ndarray: A numpy array of reshaped feature data.
        """

        reshape_data: List[np.ndarray] = []

        # Function to check for NaN or inf values
        def has_invalid_values(array: np.ndarray) -> bool:
            return np.isnan(array).any() or np.isinf(array).any()

        if config.ROI_TYPE != 'patch':
            target_shape = (config.SEQ_LENGTH, config.INPUT_SIZE)

            for feature_data in data:
                # Skip if data contains NaN or inf
                if has_invalid_values(feature_data):
                    continue

                if feature_data.shape != target_shape:
                    reshape_data.append(np.reshape(feature_data, target_shape))
                else:
                    reshape_data.append(feature_data)

        else:
            for feature_data in data:
                # Skip if data contains NaN or inf
                if has_invalid_values(feature_data):
                    continue

                if feature_data.ndim == 3: # target_shape = (patch_size, SEQ_LENGTH, INPUT_SIZE)
                    # If feature_data already has 3 dimensions, assume it is already in the correct shape
                    reshape_data.append(feature_data)
                else:
                    # Dynamically determine patch size from feature_data
                    inferred_patch_size = feature_data.shape[0]
                    target_shape = (inferred_patch_size, config.SEQ_LENGTH, config.INPUT_SIZE)
                    reshape_data.append(np.reshape(feature_data, target_shape))
        
        # Return the reshaped data if there is any, else return an empty array
        if len(reshape_data) > 0:
            return np.stack(reshape_data, axis=0)
        else:
            return np.array([])  # Return an empty array if no valid data exists
    

    def _extract_features(self, subset: str, subset_type: str) -> List[np.ndarray]:
        """
        Extracts features from the dataset based on the subset and subset type.

        Args:
            subset (str): The subset of data to load ('real' or 'deepfake').
            subset_type (str): The type of data to load ('train', 'val', 'test').

        Returns:
            List[np.ndarray]: A list of numpy arrays containing the extracted features.
        """

        feature_list: List[np.ndarray] = []
        video_output_dir: Path = self.video_output_dir / 'GoogleDFD' / subset / subset_type / config.METHOD / config.ROI_TYPE
                            
        for file in video_output_dir.glob(f'{config.FEATURE}*.npy'):
            if file.is_file() and file.name.startswith(config.FEATURE) and file.suffix == '.npy':
                try:
                    feature_data = np.load(file, allow_pickle=True)

                    # Handle the case where feature_data is a 0-d array
                    if isinstance(feature_data, np.ndarray) and feature_data.ndim == 0:
                        feature_data = feature_data.item()

                    # If feature_data is a dictionary, handle it directly
                    if isinstance(feature_data, dict):
                        if config.ROI_SUBTYPE in feature_data:
                            roi_data = feature_data[config.ROI_SUBTYPE]

                            for signal_data in roi_data:
                                for signal_type, signal in signal_data.items():
                                    if signal_type == config.SIGNAL_TYPE:
                                        feature_list.append(np.array(signal))
                                        break

                    # If feature_data is a list or an array, iterate over it
                    elif isinstance(feature_data, (list, np.ndarray)):
                        for roi in feature_data:
                            if isinstance(roi, dict) and config.ROI_SUBTYPE in roi:
                                roi_data = roi[config.ROI_SUBTYPE]
                                for signal_data in roi_data:
                                    for signal_type, signal in signal_data.items():
                                        if signal_type == config.SIGNAL_TYPE:
                                            feature_list.append(np.array(signal))
                                            break

                except Exception as e:
                    print(f"Error loading {file}: {e}")

        return feature_list
    
    def _extract_patches(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extracts patches from the input data.

        Args:
            data (List[np.ndarray]): A list of numpy arrays representing the sequence data.

        Returns:
            List[np.ndarray]: A list of numpy arrays representing the extracted patches.
        """

        return [patch for seq in data for patch in seq]

class CustomDataset(Dataset):
    """
    A custom dataset class that extends PyTorch's Dataset module, suitable for loading and preprocessing
    any generic dataset for model training and evaluation.

    This dataset class is designed to handle pairs of features and labels for tasks such as classification,
    where each element of the dataset returns a single data point and its associated label.

    Attributes:
        features (torch.Tensor): A tensor containing all the features of the dataset.
        labels (torch.Tensor): A tensor containing all the labels corresponding to the features.
    """

    def __init__(self, features: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]):
        """
        Initializes the dataset with features and labels.

        Args:
            features (np.ndarray or torch.Tensor): The features of the dataset, can be of any shape.
            labels (np.ndarray or torch.Tensor): The labels of the dataset, typically matching the first dimension of features.
        """

        self.features = torch.tensor(features, dtype=torch.float32) if isinstance(features, np.ndarray) else features
        self.labels = torch.tensor(labels, dtype=torch.long) if isinstance(labels, np.ndarray) else labels

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the feature and label at a specific index in the dataset.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature and label of the data point at the specified index.
        """

        return self.features[idx], self.labels[idx]

class DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing datasets and preparing data loaders.
    Handles operations like dataset creation, balancing, and setting up the data loaders for training, validation, and testing phases.
    """
    
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
        # self.data_balancer: Optional[DataBalancer] = None
        # self.sampler: Optional[Union[Sampler, Any]] = None
        # self.sampler_factory: Optional[ImbalancedDataSamplerFactory] = None

        self.batch_size: int = kwargs.get('batch_size', 32)
        self.num_workers: int = 0 if config.ROI_TYPE=='patch' else min(kwargs.get('num_workers', 8), mp.cpu_count() or 1)
        self.persistent_workers: bool = False if config.ROI_TYPE=='patch' else True
        self.pin_memory: bool = kwargs.get('pin_memory', True)
        self.balance_method: Optional[str] = kwargs.get('balance_method', None)
        self.sampler_method: Optional[str] = kwargs.get('sampler_method', None)
        self.pos_weights_bool: Optional[bool] = kwargs.get('pos_weights_bool', None)
        self.pos_weights: Optional[torch.Tensor] = None

        self.custom_data_loader = CustomDataLoader()
        self.logger = logging.getLogger(__name__)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setups datasets part of the data module. This method is called automatically by PyTorch Lightning
        during the start of training or validation. Here, the data is balanced according to the specified method if required.

        Args:
            stage (str, optional): Stage for which setup is being called ('fit', 'validate', 'test', 'predict').
        """

        # Load the datasets based on the current stage
        if stage == 'fit' or stage is None:
            # Load training and validation datasets
            self.train_dataset, self.val_dataset, _ = self.custom_data_loader.create_datasets(stage='fit')
            self._setup_train_sampler_and_balance()
        
        if stage == 'validate' or stage is None:
            # Load validation dataset if not already loaded
            if self.val_dataset is None:
                _, self.val_dataset, _ = self.custom_data_loader.create_datasets(stage='validate')

        if stage == 'test' or stage is None:
            # Load test dataset
            _, _, self.test_dataset = self.custom_data_loader.create_datasets(stage='test')

    def _setup_train_sampler_and_balance(self):
        """
        Balances the training data and sets up the sampler if required.
        """
        # Balance the data
        self.data_balancer = DataBalancer(self.train_dataset, self.balance_method)

        if self.balance_method:
            self.train_dataset = self.data_balancer.apply_balancing_method()

        # Calculate positive weights if applicable
        if self.pos_weights_bool:
            self.pos_weights = self.data_balancer.calculate_pos_weight()

        # Set up sampler
        if self.sampler_method:
            self.sampler_factory = ImbalancedDataSamplerFactory(self.train_dataset, self.sampler_method, self.batch_size)
            self.sampler = self.sampler_factory.get_sampler()
    
    def train_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader for training data.
        """
        if self.train_loader is None:
            if self.sampler:
                if isinstance(self.sampler, (BalancedBatchSampler, WeightedBatchSampler)):
                    self.train_loader = DataLoader(
                        self.train_dataset,
                        batch_sampler=self.sampler,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        persistent_workers=self.persistent_workers,
                        worker_init_fn=self.worker_init_fn
                    )
                    # for batch in self.train_loader:
                    #     features, labels = batch
                    #     print("Batch labels:", labels)
                    #     break

                else:
                    self.train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=self.batch_size,
                        sampler=self.sampler,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        persistent_workers=self.persistent_workers,
                        worker_init_fn=self.worker_init_fn
                    )
            else:
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    worker_init_fn=self.worker_init_fn
                )
        return self.train_loader
        
    def val_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader for validation data.
        """
        if self.val_loader is None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                worker_init_fn=self.worker_init_fn
            )
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the testing dataset.

        Returns:
            DataLoader: The DataLoader for testing data.
        """
        if self.val_loader is None:
            self.val_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                worker_init_fn=self.worker_init_fn
            )
        return self.val_loader

    def worker_init_fn(self, worker_id: int):
        """
        Initializes the worker function for data loaders to ensure different random seed for each worker.
        """
        base_seed = torch.initial_seed() % 2**32
        np.random.seed(base_seed + worker_id)
        torch.manual_seed(base_seed + worker_id)

    # def cleanup(self):
    #     """
    #     Cleans up the DataModule by clearing datasets, dataloaders, and other resources.
    #     """
    #     self.train_dataset = None
    #     self.val_dataset = None
    #     self.test_dataset = None
    #     self.train_loader = None
    #     self.val_loader = None
    #     self.test_loader = None

    #     # Free GPU memory if applicable
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     self.logger.info("DataModule cleanup completed.")


class ImbalancedDataSamplerFactory:
    """
    A factory class for creating samplers to handle imbalanced datasets in PyTorch. This class
    provides methods to generate different types of samplers based on the specified sampling strategy.

    Attributes:
        dataset (Dataset): The dataset for which the sampler is being created.
        sampling_strategy (str): The strategy used for sampling ('balanced', 'weighted', 'balanced_batch', 'weighted_batch').
        batch_size (int): The batch size to be used by the sampler.
        labels (torch.Tensor): The labels of the dataset.
        class_count (torch.Tensor): The count of samples for each class in the dataset.
        num_samples (int): The total number of samples in the dataset.
        n_classes (int): The number of unique classes in the dataset.
    """

    def __init__(self, dataset: Dataset, sampling_strategy: str = 'balanced', batch_size: int = 32):
        self.dataset: Dataset = dataset
        self.sampling_strategy: str = sampling_strategy
        self.sampler: Optional[Union[BatchSampler, WeightedRandomSampler, Any]] = None
        self.batch_size: int = batch_size
        self.labels: torch.Tensor = self._get_labels()
        self.class_count: torch.Tensor = self._get_class_count()
        self.num_samples: int = len(self.labels)
        self.n_classes: int = len(self.class_count)
        
        # print(f"ImbalancedDataSamplerFactory initialized with {self.num_samples} samples and {self.n_classes} classes")
        self._validate_dataset()

    def _validate_dataset(self):
        """
        Validates the dataset to ensure it has the required attributes and valid data.

        Raises:
            AttributeError: If the dataset does not have a __len__ method or 'labels' attribute.
            ValueError: If the dataset length does not match the number of labels or if there are fewer than 2 classes.
        """

        if not hasattr(self.dataset, '__len__'):
            raise AttributeError("Dataset must have a __len__ method")
        if len(self.dataset) != self.num_samples:
            raise ValueError(f"Dataset length ({len(self.dataset)}) does not match number of labels ({self.num_samples})")
        if self.n_classes < 2:
            raise ValueError(f"Dataset must have at least 2 classes, but found {self.n_classes}")

    def _get_labels(self) -> torch.Tensor:
        """
        Retrieves the labels from the dataset.

        Returns:
            torch.Tensor: A 1D tensor containing the labels of the dataset.

        Raises:
            AttributeError: If the dataset does not have a 'labels' attribute.
            TypeError: If the labels are not a torch.Tensor.
            ValueError: If the labels tensor is not 1-dimensional.
        """

        if not hasattr(self.dataset, 'labels'):
            raise AttributeError("Dataset has no 'labels' attribute")
        labels = self.dataset.labels
        if not isinstance(labels, torch.Tensor):
            raise TypeError("Labels must be a torch.Tensor")
        if labels.dim() != 1:
            raise ValueError("Labels must be a 1-dimensional tensor")
        return labels
    
    def _get_class_count(self) -> torch.Tensor:
        """
        Calculates the number of samples for each class in the dataset.

        Returns:
            torch.Tensor: A tensor containing the count of samples for each class.

        Raises:
            ValueError: If any class in the dataset has zero samples.
        """

        class_count = torch.bincount(self.labels)
        if torch.any(class_count == 0):
            raise ValueError("Found classes with zero samples")
        return class_count

    def _compute_weights(self) -> torch.Tensor:
        """
        Computes the sampling weights for each class based on the sampling strategy.

        Returns:
            torch.Tensor: A tensor containing the weights for each class.

        Raises:
            ValueError: If the sampling strategy is invalid.
        """

        if self.sampling_strategy in ['balanced', 'balanced_batch']:
            # Equal probability for each class
            weights = self.num_samples / (self.n_classes * self.class_count.float())
        elif self.sampling_strategy in ['weighted', 'weighted_batch']:
            # Higher weight for rare classes, but not equal probability
            weights = 1.0 / self.class_count.float()
            # Normalize weight for classes
            weights = weights / weights.sum()
        else:
            raise ValueError("Invalid sampling strategy. Choose 'balanced', 'weighted', 'balanced_batch', or 'weighted_batch'.")
        
        return weights

    def get_sampler(self) -> Sampler:
        """
        Creates and returns the appropriate sampler based on the sampling strategy.

        Returns:
            Sampler: The sampler to be used for data loading.

        Raises:
            ValueError: If the dataset size does not match the expected number of samples or if the sampling strategy is invalid.
        """

        if self.num_samples != len(self.dataset):
            raise ValueError(f"Dataset size mismatch. Expected {self.num_samples}, but got {len(self.dataset)}")
        
        if self.sampling_strategy in ['balanced', 'weighted']:
            return self._get_weighted_random_sampler()
        elif self.sampling_strategy == 'balanced_batch':
            return self._get_balanced_batch_sampler()
        elif self.sampling_strategy == 'weighted_batch':
            return self._get_weighted_batch_sampler()
        else:
            raise ValueError("Invalid sampling strategy.")

    def _get_weighted_random_sampler(self) -> WeightedRandomSampler:
        """
        Creates a WeightedRandomSampler based on the computed class weights.

        Returns:
            WeightedRandomSampler: A sampler that samples elements according to the specified weights.
        """

        class_weights = self._compute_weights()
        sample_weights = class_weights[self.labels]
        return WeightedRandomSampler(sample_weights, num_samples=self.num_samples, replacement=True)

    def _get_balanced_batch_sampler(self) -> BatchSampler:
        """
        Creates a BalancedBatchSampler for balanced sampling within batches.

        Returns:
            BalancedBatchSampler: A balanced batch sampler instance.
        """

        class_weights = self._compute_weights()
        class_idxs = [torch.where(self.labels == i)[0] for i in range(self.n_classes)]
        return BalancedBatchSampler(class_weights.tolist(), class_idxs, self.batch_size)

    def _get_weighted_batch_sampler(self) -> BatchSampler:
        """
        Creates a WeightedBatchSampler for weighted sampling within batches.

        Returns:
            WeightedBatchSampler: A weighted batch sampler instance.
        """

        class_weights = self._compute_weights()
        class_idxs = [torch.where(self.labels == i)[0] for i in range(self.n_classes)]
        return WeightedBatchSampler(class_weights.tolist(), class_idxs, self.batch_size)

class BalancedBatchSampler(BatchSampler):
    """
    A batch sampler that generates balanced batches of data, ensuring that each batch contains samples
    from all classes. This is useful for handling imbalanced datasets during training.

    Attributes:
        class_weights (List[float]): A list of weights for each class, used to balance the sampling probability.
        class_idxs (List[List[int]]): A list of indices for each class.
        batch_size (int): The number of samples in each batch.
        n_classes (int): The number of unique classes in the dataset.
        min_samples_per_class (int): The minimum number of samples to draw from each class for a batch.
    """
    
    def __init__(self, class_weights: List[float], class_idxs: List[List[int]], batch_size: int):
        super().__init__(None, batch_size, drop_last=False)
        self.class_weights = class_weights
        self.class_idxs = class_idxs
        self.batch_size = batch_size
        self.n_classes = len(class_idxs)
        self.min_samples_per_class = max(1, batch_size // 2 * self.n_classes) # *2 ensures flexebility
        
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validates the input parameters to ensure that they are consistent with the requirements of the sampler.

        Raises:
            ValueError: If batch size is less than the number of classes, or if the number of class weights
                        does not match the number of classes, or if any class has no samples.
        """

        if self.batch_size < self.n_classes:
            raise ValueError(f"Batch size ({self.batch_size}) must be at least as large as the number of classes ({self.n_classes})")
        if len(self.class_weights) != self.n_classes:
            raise ValueError(f"Number of class weights ({len(self.class_weights)}) does not match number of classes ({self.n_classes})")
        if any(len(idxs) == 0 for idxs in self.class_idxs):
            raise ValueError("Found classes with no samples")

    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns an iterator that yields balanced batches of indices.

        Yields:
            Iterator[List[int]]: An iterator that yields a list of indices representing a balanced batch.
        """

        while True:
            batch: List[int] = []
            # Ensure that each class is represented with a minimum number of samples in the batch
            for class_idx in range(self.n_classes):
                if len(self.class_idxs[class_idx]) < self.min_samples_per_class:
                    raise RuntimeError(f"Not enough samples in class {class_idx} for balanced sampling")
                batch.extend(random.choices(self.class_idxs[class_idx], k=self.min_samples_per_class))
            
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                # Fill the remaining part of the batch with random samples but keep the class balance
                all_idxs = [idx for idxs in self.class_idxs for idx in idxs]
                weights = []
                for i, idxs in enumerate(self.class_idxs):
                    weights.extend([self.class_weights[i]] * len(idxs))
                
                # Sample the remaining indices while preserving class balance
                batch.extend(random.choices(all_idxs, weights=weights, k=remaining))
            random.shuffle(batch)  # Shuffle the batch to mix class samples
            yield batch

    def __len__(self) -> int:
        """
        Returns the total number of batches that can be generated by the sampler.

        Returns:
            int: The total number of batches.
        """

        return (sum(len(idxs) for idxs in self.class_idxs) + self.batch_size - 1) // self.batch_size

class WeightedBatchSampler(BatchSampler):
    """
    A batch sampler that generates weighted batches of data, allowing for classes with fewer samples
    to be sampled more frequently based on their assigned weights. This is useful for handling imbalanced datasets.

    Attributes:
        class_weights (List[float]): A list of weights for each class, used to adjust the sampling probability.
        class_idxs (List[List[int]]): A list of indices for each class.
        batch_size (int): The number of samples in each batch.
        sample_idxs (List[int]): A flattened list of all sample indices.
        sample_weights (List[float]): A list of weights corresponding to each sample in `sample_idxs`.
    """

    def __init__(self, class_weights: List[float], class_idxs: List[List[int]], batch_size: int):
        super().__init__(None, batch_size, drop_last=False)
        self.class_weights = class_weights
        self.class_idxs = class_idxs
        self.batch_size = batch_size
        self.sample_idxs = [idx for idxs in class_idxs for idx in idxs]
        self.sample_weights = [weight for c, weight in enumerate(class_weights) for _ in class_idxs[c]]

    def _validate_inputs(self):
        """
        Validates the input parameters to ensure they are consistent with the requirements of the sampler.

        Raises:
            ValueError: If the batch size is not positive, if the number of class weights does not match the number
                        of class indices, if any class has no samples, or if there is a mismatch between
                        the number of sample indices and weights.
        """

        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if len(self.class_weights) != len(self.class_idxs):
            raise ValueError(f"Number of class weights ({len(self.class_weights)}) does not match number of class indices ({len(self.class_idxs)})")
        if any(len(idxs) == 0 for idxs in self.class_idxs):
            raise ValueError("Found classes with no samples")
        if len(self.sample_idxs) == 0:
            raise ValueError("No samples to sample from")
        if len(self.sample_weights) != len(self.sample_idxs):
            raise ValueError("Mismatch between number of sample indices and weights")

    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns an iterator that yields weighted batches of indices.

        Yields:
            Iterator[List[int]]: An iterator that yields a list of indices representing a weighted batch.
        """

        while True:
            yield random.choices(self.sample_idxs, weights=self.sample_weights, k=self.batch_size)

    def __len__(self) -> int:
        """
        Returns the total number of batches that can be generated by the sampler.

        Returns:
            int: The total number of batches.
        """

        return (len(self.sample_idxs) + self.batch_size - 1) // self.batch_size

class DataBalancer:
    """
    A class for balancing datasets by applying various sampling methods such as oversampling, undersampling, or SMOTE.
    The class can also calculate class weights for use in loss functions to handle class imbalance.

    Attributes:
        balance_method (str): The method used to balance the dataset ('smote', 'oversample', 'undersample').
        dataset (Dataset): The dataset to be balanced.
        features (torch.Tensor): The features of the dataset.
        labels (torch.Tensor): The labels corresponding to the features.
        class_count (torch.Tensor): The count of samples in each class.
        logger (logging.Logger): Logger for logging information and errors.
    """

    def __init__(self, dataset: Dataset, balance_method: str):
        self.logger = logging.getLogger(__name__)
        self.balance_method = balance_method
        self.dataset = dataset
        self.features: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.class_count: Optional[torch.Tensor] = None
        
        # Initialize features, labels, and class count
        self._initialize()

    def _initialize(self):
        """
        Initializes features, labels, and class count attributes.
        """
        
        try:
            self.features = self._get_features()
            self.labels = self._get_labels()
            self.class_count = self._get_class_count()
            self._validate_dataset()
        except Exception as e:
            self.logger.error("Error during DataBalancer initialization: %s", e)
            raise

    def _validate_dataset(self):
        """
        Validates the dataset to ensure it has the required attributes and valid data.

        Raises:
            AttributeError: If the dataset does not have a __len__ method or 'labels' attribute.
            ValueError: If the dataset length does not match the number of labels or if there are fewer than 2 classes.
        """

        if not hasattr(self.dataset, '__len__'):
            raise AttributeError("Dataset must have a __len__ method")
        if self.labels is None:
            raise ValueError("Labels are not initialized or are None.")
        if len(self.dataset) != len(self.labels):
            raise ValueError(f"Dataset length ({len(self.dataset)}) does not match number of labels ({len(self.labels)})")
        if self.class_count is None or len(self.class_count) < 2:
            raise ValueError(f"Dataset must have at least 2 classes, but found {len(self.class_count)}")
        
    def calculate_pos_weight(self) -> torch.Tensor:
        """
        Calculates the weights for each class based on the number of samples in each class, which can be used
        to balance the dataset during training by adjusting the loss function.

        Returns:
            torch.Tensor: Tensor containing the weight for each class.
        """
        
        if len(self.class_count) > 1:
            return self.class_count[0].float() / self.class_count[1].float()
        return torch.tensor(1.0)  # Default weight if only one class

    def apply_balancing_method(self) -> Dataset:
        """
        Applies the specified balancing method to the dataset and returns a new balanced dataset.
        """
        
        try:
            if self.balance_method == 'smote':
                return self.smote_data()
            elif self.balance_method == 'oversample':
                return self.oversample_data()
            elif self.balance_method == 'undersample':
                return self.undersample_data()
            else:
                raise ValueError(f"Invalid balancing method: {self.balance_method}")
        except Exception as e:
            self.logger.error("Error during balancing: %s", e)
            raise
        
    def _get_labels(self) -> torch.Tensor:
        """
        Retrieves the labels from the dataset.

        Returns:
            torch.Tensor: A tensor containing the labels of the dataset.

        Raises:
            AttributeError: If the dataset does not have a 'labels' attribute.
        """

        if hasattr(self.dataset, 'labels'):
            return self.dataset.labels
        else:
            raise AttributeError("Dataset has no 'labels' attribute")
        
    def _get_features(self) -> torch.Tensor:
        """
        Retrieves the features from the dataset.

        Returns:
            torch.Tensor: A tensor containing the features of the dataset.

        Raises:
            AttributeError: If the dataset does not have a 'features' attribute.
        """

        if hasattr(self.dataset, 'features'):
            return self.dataset.features
        else:
            raise AttributeError("Dataset has no 'features' attribute")

    def _get_class_count(self) -> torch.Tensor:
        """
        Calculates the number of samples for each class in the dataset.

        Returns:
            torch.Tensor: A tensor containing the count of samples for each class.
        """

        return torch.bincount(self.labels)

    def _get_class_count_dict(self, labels: torch.Tensor) -> dict:
        """
        Creates a dictionary representing the number of samples for each class.

        Args:
            labels (torch.Tensor): A tensor containing the labels of the dataset.

        Returns:
            dict: A dictionary where keys are class indices and values are the number of samples in each class.
        """

        class_count = torch.bincount(labels)
        class_count_dict = {i: count.item() for i, count in enumerate(class_count)}
        return class_count_dict

    def oversample_data(self) -> CustomDataset:
        """
        Applies oversampling to the dataset to balance the class distribution.

        Returns:
            CustomDataset: The oversampled dataset.
        """

        max_class_samples = torch.max(self.class_count)
        resampled_features, resampled_labels = [], []

        for class_index in torch.unique(self.labels):
            class_mask = (self.labels == class_index)
            class_features = self.features[class_mask]
            class_labels = self.labels[class_mask]

            num_samples = self.class_count[class_index]
            sampled_indices = torch.randint(0, num_samples.item(), (max_class_samples.item() - num_samples.item(),))
            resampled_features.append(class_features[sampled_indices])
            resampled_labels.append(class_labels[sampled_indices])

        balanced_features = torch.cat(resampled_features + [self.features])
        balanced_labels = torch.cat(resampled_labels + [self.labels])
        
        del resampled_features, resampled_labels  # Free memory
        return CustomDataset(balanced_features, balanced_labels)

    def undersample_data(self) -> CustomDataset:
        """
        Applies undersampling to the dataset to balance the class distribution.

        Returns:
            Any: The undersampled dataset.
        """

        min_class_samples = torch.min(self.class_count)
        resampled_features, resampled_labels = [], []

        for class_index in torch.unique(self.labels):
            class_mask = (self.labels == class_index)
            class_features = self.features[class_mask]
            class_labels = self.labels[class_mask]

            resampled_features.append(class_features[:min_class_samples])
            resampled_labels.append(class_labels[:min_class_samples])

        balanced_features = torch.cat(resampled_features)
        balanced_labels = torch.cat(resampled_labels)

        del resampled_features, resampled_labels  # Free memory
        return CustomDataset(balanced_features, balanced_labels)

    def smote_data(self) -> CustomDataset:
        """
        Applies SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.

        Returns:
            Any: The dataset with synthetic samples added to balance the class distribution.

        Raises:
            ValueError: If the input shape is unexpected.
        """

        # Determine the shape of the input
        original_feature_shape = self.features.ndim
        if original_feature_shape == 4: # For patch data
            num_samples, num_patches, seq_length, num_features = self.features.shape
        elif original_feature_shape == 3: # For non-patch data
            num_samples, seq_length, num_features = self.features.shape
        else:
            raise ValueError(f"Unexpected input shape: {original_feature_shape}")

        # Reshape the array to 2 dimensions
        features_reshaped = self.features.reshape(num_samples, -1)

        # Apply SMOTE
        smote = SMOTE(k_neighbors=100, random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features_reshaped, self.labels)

        # Reshape the resampled features back to original dimensions
        if original_feature_shape == 4:
            features_resampled_reshaped = features_resampled.reshape(-1, num_patches, seq_length, num_features)
        elif original_feature_shape == 3:
            features_resampled_reshaped = features_resampled.reshape(-1, seq_length, num_features)

        del features_reshaped  # Free memory
        return CustomDataset(features_resampled_reshaped, labels_resampled)
