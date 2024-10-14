"""
This module handles the processing of video files for face detection, mask creation,
and pulse signal analysis. It includes functionalities for splitting datasets, 
logging, and multiprocessing.

Classes:
    DataSplitter: Splits a list of video paths into training, validation, and test sets.

Functions:
    get_memory_usage: Returns the current memory usage of the process.
    progresser: Processes video batches and saves the results.
    create_batches: Creates batches from a list of videos.
    setup_logging: Sets up the logging configuration.
    process_dataset: Processes videos in the given dataset.
    process_videos_with_multiprocessing: Processes videos using multiprocessing.
    process_videos_without_multiprocessing: Processes videos without using multiprocessing.

Example usage:
    To use this module, ensure that the path_config module is properly configured and 
    contains the necessary dataset paths. Then, run the script as follows:

    if __name__ == "__main__":
        data_splitter = DataSplitter()
        
        logger = setup_logging()
        logger.info("Starting main processing")

        args_list = []

        for dataset in path_config.DATASETS:
            dataset_args = process_dataset(dataset, data_splitter, logger)
            args_list.extend(dataset_args)
        
        USE_MULTIPROCESSING = True # Set to False if multiprocessing is not desired

        if use_multiprocessing:
            process_videos_with_multiprocessing(args_list, logger)
        else:
            process_videos_without_multiprocessing(args_list)
"""

import os
import gc
import warnings
import logging
from pathlib import Path
from typing import List, Tuple, Union
import random
from multiprocessing import Pool
import psutil
from tqdm.auto import tqdm
from colorama import Fore, Style

# logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Disable oneDNN optimizations

# Set logging level to ERROR to suppress info/debug logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from rppg_facepatches.video_processor import mp_config
from rppg_facepatches.video_processor import path_config
from rppg_facepatches.video_processor.video_handler import VideoHandler
from rppg_facepatches.video_processor.face_detector import FaceDetector

class DataSplitter:
    """
    A class to split a list of video paths into training, validation, and test sets.

    Attributes:
        val_percent (float): Percentage of videos for the validation set.
        test_percent (float): Percentage of videos for the test set.
        random_seed (int): Seed for random number generation for reproducibility.

    Methods:
        split_videos(video_list: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
            Splits video paths into training, validation, and test sets.

        check_for_overlaps(train_videos: List[Path], val_videos: List[Path], test_videos: List[Path]) -> None:
            Checks for overlaps between the split datasets and raises an error if overlaps exist.
    """

    def __init__(self, val_percent: float = 0.2, test_percent: float = 0.2, random_seed: int = 42):
        """
        Initializes the DataSplitter with the specified validation and test percentages, and a random seed.

        Args:
            val_percent (float): Percentage of videos for the validation set. Defaults to 0.2.
            test_percent (float): Percentage of videos for the test set. Defaults to 0.2.
            random_seed (int): Seed for random number generation for reproducibility. Defaults to 42.
        """
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.random_seed = random_seed

    def split_videos(self, video_list: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Splits the given list of video paths into training, validation, and test sets.

        Args:
            video_list (List[Path]): The list of video file paths to be split.

        Returns:
            Tuple[List[Path], List[Path], List[Path]]: A tuple containing three lists:
                - train_videos: List of video paths for the training set.
                - val_videos: List of video paths for the validation set.
                - test_videos: List of video paths for the test set.

        Raises:
            ValueError: If there is any overlap between the train, validation, and test sets.
        """
        random.seed(self.random_seed)
        amount = len(video_list)
        test_amount = (int(amount * self.test_percent) if self.test_percent is not None else 0)
        val_amount = (int(amount * self.val_percent) if self.val_percent is not None else 0)
        train_amount = amount - test_amount - val_amount

        random.shuffle(video_list)

        train_videos = video_list[:train_amount]
        val_videos = video_list[train_amount : train_amount + val_amount]
        test_videos = video_list[train_amount + val_amount :]

        self.check_for_overlaps(train_videos, val_videos, test_videos)

        return train_videos, val_videos, test_videos
    
    def check_for_overlaps(self, train_videos, val_videos, test_videos):
        """
        Checks for overlaps between the training, validation, and test sets.

        Args:
            train_videos (List[Path]): Video paths for the training set.
            val_videos (List[Path]): Video paths for the validation set.
            test_videos (List[Path]): Video paths for the test set.

        Raises:
            ValueError: If overlaps are detected between any of the sets.
        """
        train_set = set(train_videos)
        val_set = set(val_videos)
        test_set = set(test_videos)

        if train_set & val_set:
            raise ValueError("Overlap detected between training and validation sets.")
        if train_set & test_set:
            raise ValueError("Overlap detected between training and test sets.")
        if val_set & test_set:
            raise ValueError("Overlap detected between validation and test sets.")


def get_memory_usage() -> int:
    """
    Get the current memory usage of the process.

    Returns:
        int: The Resident Set Size (RSS) in bytes.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss  # Resident Set Size (RSS) in bytes

def create_batches(videos: Union[List[Path], Path], batch_size: int, use_subsets: bool) -> List[List[Tuple[Path, int]]]:
    """
    Creates batches from a list of videos.

    Args:
        videos (List[Path]): List of video file paths.
        batch_size (int): The size of each batch.
        use_subsets (bool): If True, creates one batch for the entire subset.

    Returns:
        List[List[Tuple[Path, int]]]: List of batches, each containing tuples of video file paths and indices.
    """
    if use_subsets:
        if isinstance(videos, List):
            batches = [[(video_path, idx) for idx, video_path in enumerate(videos)]]
        elif isinstance(videos, Path):
            batches = [[(video_path, idx) for idx, video_path in enumerate([videos])]]
    else:
        if isinstance(videos, List):
            batches = [[(video, idx) for idx, video in enumerate(videos[i:i + batch_size], start=i)] for i in range(0, len(videos), batch_size)]
        elif isinstance(videos, Path):
            batches = [[(video_path, idx) for idx, video_path in enumerate([videos])]]
    return batches

def setup_logging() -> logging.Logger:
    """
    Sets up the logging configuration.

    Args:
        project_path (Path): The path to the project directory.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_directory = path_config.PROJECT_PATH / "logs"
    log_directory.mkdir(exist_ok=True)
    main_log_file = str(log_directory / "main.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=main_log_file,
        filemode="a",
    )
    return logging.getLogger(__name__)

def preprocess_dataset(
        dataset: str,
        data_splitter: DataSplitter,
        logger: logging.Logger,
        use_subset: bool,
        batch_size: int = 5,
    ) -> List[Tuple[List[Tuple[Path, int]], str, str, str]]:
    """
    Processes the videos in the given dataset, splitting them into train, validation, and test sets,
    and then processing and saving the video batches.

    Args:
        dataset (str): The dataset name.
        data_splitter (DataSplitter): An instance of DataSplitter.
        logger (logging.Logger): Logger instance for logging.

    Returns:
        List[Tuple[List[Tuple[str, int]], str, str, str]]: List of arguments for processing video batches.
    """
    path_config.DATASET = dataset
    path_config.BASE_DATASET_PATH, path_config.REAL_DATASET_PATH, path_config.DEEPFAKE_DATASET_PATH = path_config.setup_dataset_paths() # path_config.DATASET

    # Process real videos
    real_videos = list(path_config.REAL_DATASET_PATH.glob("*.mp4"))
    logger.info("Number of real videos: %d", len(real_videos))
    real_train_videos, real_val_videos, real_test_videos = data_splitter.split_videos(real_videos)
    logger.info(
        "Real videos split - Train: %d, Test: %d, Val: %d",
        len(real_train_videos),
        len(real_test_videos),
        len(real_val_videos),
    )

    # Process deepfake videos
    deepfake_videos = list(path_config.DEEPFAKE_DATASET_PATH.glob("*.mp4"))
    logger.info("Number of deepfake videos: %d", len(deepfake_videos))
    deepfake_train_videos, deepfake_val_videos, deepfake_test_videos = data_splitter.split_videos(deepfake_videos)
    logger.info(
        "Deepfake videos split - Train: %d, Test: %d, Val: %d",
        len(deepfake_train_videos),
        len(deepfake_test_videos),
        len(deepfake_val_videos),
    )

    video_sets = [
        # (deepfake_test_videos[1], "deepfake/test"),
        # (deepfake_train_videos[1], "deepfake/train"),
        # (deepfake_val_videos[1], "deepfake/val"),
        (real_test_videos[1], "real/test"),
        (real_val_videos, "real/val"),
        (real_train_videos, "real/train"),
    ]
    methods = ["rgb"] # "orgb"
    roi_types = ['patch'] #"background", "mask", patch

    args = [
        (video_batch, subfolder, method, roi_type)
        for method in methods
        for roi_type in roi_types
        for videos, subfolder in video_sets
        for video_batch in create_batches(videos, batch_size, use_subsets=use_subset)
    ]

    # Log the number of videos per subfolder
    # for videos, subfolder in video_sets:
    #     logger.info(f"Subfolder {subfolder}: {len(videos)}")

    return args

def process_video(args):
    """Process a single video."""
    video_path, index, subfolder, method, roi_type = args

    # Process the video
    video_handler = VideoHandler()
    face_detector = FaceDetector()
    video_handler.process_and_save_video(face_detector, video_path, index, subfolder, method, roi_type)
    face_detector.close()

def process_videos_with_multiprocessing(args_list: List[Tuple[List[Tuple[Path, int]], str, str, str]]):
    """
    Processes videos using multiprocessing.

    Args:
        args_list (List[Tuple[List[Tuple[Path, int]], str, str, str]]): List of arguments for processing video batches.
    """
    # Calculate the total number of videos
    total_videos = sum(len(video_batch) for video_batch, _, _, _ in args_list)
    num_processes = mp_config.NUM_PROCESSES

    # Create the main progress bar for all videos
    with tqdm(total=total_videos, desc=Fore.RED + "Total progress" + Style.RESET_ALL, position=0, colour="red") as total_progress:
        for batch_index, (video_batch, subfolder, method, roi_type) in enumerate(tqdm(args_list, desc=Fore.GREEN + "Processing video batches" + Style.RESET_ALL, position=1, colour="green")):
            # Create a nested progress bar for each video batch
            with tqdm(total=len(video_batch), desc=Fore.BLUE + f"Batch {batch_index+1}/{len(args_list)} ({method}, {subfolder}, {roi_type}) progress" + Style.RESET_ALL, position=2, colour="blue", leave=False) as batch_progress:
                 # Create a pool of workers
                with Pool(processes=num_processes) as pool:
                    # Prepare arguments for pool.imap_unordered
                    args_for_pool = [(video_path, index, subfolder, method, roi_type) for video_path, index in video_batch]

                    for _ in pool.imap_unordered(process_video, args_for_pool):
                        total_progress.update(1)
                        batch_progress.update(1)

def process_videos_without_multiprocessing(args_list: List[Tuple[List[Tuple[Path, int]], str, str, str]]):
    """
    Processes videos without using multiprocessing.

    Args:
        args_list (Union[List[Tuple[List[Tuple[Path, int]], str, str, str]], List[Tuple[Path, int, str, str, str]]]):
        List of arguments for processing video batches or individual videos.
    """
    # Calculate the total number of videos
    total_videos = sum(len(video_batch) for video_batch, _, _, _ in args_list)

    video_handler = VideoHandler()

    # Create the main progress bar for all videos
    with tqdm(total=total_videos, desc=Fore.BLUE + "Total progress" + Style.RESET_ALL, position=0, unit='video', dynamic_ncols=True, colour='blue') as total_progress:
        for batch_index, (video_batch, subfolder, method, roi_type) in enumerate(tqdm(args_list, desc=Fore.GREEN + "Processing video batches" + Style.RESET_ALL, position=1, unit='batch', dynamic_ncols=True, colour='green')):

            # Create a nested progress bar for each video batch
            with tqdm(total=len(video_batch), desc=f"Batch {batch_index+1}/{len(args_list)} ({method}, {subfolder}, {roi_type}) progress", position=2, leave=False, unit='video', dynamic_ncols=True) as batch_progress:
                for video_path, index in video_batch:
                    face_detector = FaceDetector()
                    video_handler.process_and_save_video(face_detector, video_path, index, subfolder, method, roi_type)
                    face_detector.close()
                    batch_progress.update(1)
                    total_progress.update(1)


if __name__ == "__main__":
    gc.collect()
    data_splitter = DataSplitter()
    
    logger = setup_logging()
    logger.info("Starting main processing")

    args_list = []
    for dataset in path_config.DATASETS:
        dataset_args = preprocess_dataset(dataset, data_splitter, logger, use_subset=True)
        args_list.extend(dataset_args)

    mp_config.USE_MULTIPROCESSING = False # Set to False if multiprocessing is not desired

    if mp_config.USE_MULTIPROCESSING:
        process_videos_with_multiprocessing(args_list)
    else:
        process_videos_without_multiprocessing(args_list)