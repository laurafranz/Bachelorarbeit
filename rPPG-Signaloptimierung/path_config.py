"""
This module defines configurations and paths for datasets and projects used in the processing
of video files for face tracking, mask creation, and pulse signal analysis. It provides utilities
for setting up and updating paths based on the specified dataset and processing parameters.

Global Variables:
    DATASETS (List[str]): List of supported datasets.
    DATASET (str): Name of the dataset currently being used.
    SUBFOLDER (str): Subfolder name within the dataset for organizing processed data.
    METHOD (str): Processing method used (e.g., 'orgb', 'rgb').
    ROI_TYPE (str): Region of interest type (e.g., 'mask', 'background', 'patch').
    VIDEO_INDEX (int): Index of the current video being processed.
    BASE_DATASET_PATH (Path): Base path for the dataset, set during setup.
    REAL_DATASET_PATH (Path): Path for real video sequences, set during setup.
    DEEPFAKE_DATASET_PATH (Path): Path for deepfake video sequences, set during setup.
    VIDEO_FILE_PATH (Path): Path to the current video file.
    PROJECT_PATH (Path): Path to the current project directory.
    COMMON_PATH_SUFFIX (Path): Common suffix for constructing output paths.
    VIDEO_OUTPUT_DIR (Path): Directory for storing processed video output files.
    SNR_OPTIMIZER_OUTPUT_DIR (Path): Directory for storing SNR optimizer output files.
    SIGNAL_ANALYZER_OUTPUT_DIR (Path): Directory for storing signal analyzer output files.

Functions:
    setup_dataset_paths() -> Tuple[Path, Path, Path]:
        Sets up the paths for the specified dataset based on the dataset's name.
        Returns the base, real, and deepfake paths for the dataset.

    update_paths() -> Tuple[Path, Path, Path]:
        Updates and returns the global output paths for video processing outputs,
        using current global variables like dataset name and video index.

Usage:
    This module should be imported to manage dataset configurations and paths for processing tasks.
    Call `setup_dataset_paths()` at the start of the processing to initialize paths.
    Use `update_paths()` to refresh and get the current output directories based on processing parameters.
"""

from pathlib import Path
from typing import Tuple


# Dataset configurations
DATASETS = ["GoogleDFD"]  # , "FaceForensics++/Deepfakes", "CelebDFv2"
DATASET = "default_dataset" # 'GoogleDFD
SUBFOLDER = "default_subfolder" # 'real/train', 'real/val', 'real/test', 'deepfake/train', 'deepfake/val', 'deepfake/test'
METHOD = "default_method" # 'rgb', 'orgb'
ROI_TYPE = "default_roi_type" # 'background', 'mask', 'patch'
VIDEO_INDEX = 0  # Default video index

# Dataset paths
BASE_DATASET_PATH = Path("default_path")
REAL_DATASET_PATH = BASE_DATASET_PATH / "default_path"
DEEPFAKE_DATASET_PATH = BASE_DATASET_PATH / "default_path"
VIDEO_FILE_PATH = Path("default_path")

# Project paths
PROJECT_PATH = Path(__file__).resolve().parent

# Common path part
COMMON_PATH_SUFFIX = (
    Path(DATASET)
    / SUBFOLDER
    / METHOD
    / ROI_TYPE
    / f"{VIDEO_INDEX}_{VIDEO_FILE_PATH.stem}"
)

OUTPUT_PATH = PROJECT_PATH / 'output'
VIDEO_OUTPUT_DIR = OUTPUT_PATH / "pulse_signals" / COMMON_PATH_SUFFIX.parent
PULSE_SIGNALS_FILE_NAME = f'pulse_signals_{VIDEO_INDEX}_{VIDEO_FILE_PATH.stem}.npy'

SIGNAL_VISUALIZER_OUTPUT_DIR = OUTPUT_PATH / "video_visualizer" / COMMON_PATH_SUFFIX

SNR_OPTIMIZER_OUTPUT_DIR = OUTPUT_PATH / 'snr_optimizer' / COMMON_PATH_SUFFIX
SNR_ANALYZER_OUTPUT_DIR = OUTPUT_PATH / 'snr_analyzer' / DATASET / SUBFOLDER / METHOD / ROI_TYPE

SIGNAL_ANALYZER_OUTPUT_DIR = OUTPUT_PATH / "signal_analyzer" / COMMON_PATH_SUFFIX
MASK_VISUALIZER_OUTPUT_DIR = OUTPUT_PATH / "mask_visualizer" / COMMON_PATH_SUFFIX

PATCH_WEIGHTS_OPTIMIZER_OUTPUT_DIR = OUTPUT_PATH / 'patch_weights_optimizer' / VIDEO_FILE_PATH.stem
PATCH_WEIGHTS_ANALYZER_OUTPUT_DIR = OUTPUT_PATH / 'patch_weights_analyzer'


def setup_dataset_paths() -> Tuple[Path, Path, Path]:
    """
    Sets up the paths for the specified dataset.

    Returns:
        Tuple[Path, Path, Path]: Paths to the dataset, real videos, and deepfake videos.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if DATASET.startswith("FaceForensics++"):
        BASE_DATASET_PATH = Path("/datasets/BAFranz/datasets/FaceForensics++")
        REAL_DATASET_PATH = BASE_DATASET_PATH / "original_sequences/actors/c23/videos"
        if DATASET.endswith("Deepfakes"):
            DEEPFAKE_DATASET_PATH = (
                BASE_DATASET_PATH / "manipulated_sequences/Deepfakes/c23/videos"
            )
        elif DATASET.endswith("Face2Face"):
            DEEPFAKE_DATASET_PATH = (
                BASE_DATASET_PATH / "manipulated_sequences/Face2Face/c23/videos"
            )
        elif DATASET.endswith("FaceShift"):
            DEEPFAKE_DATASET_PATH = (
                BASE_DATASET_PATH / "manipulated_sequences/FaceShift/c23/videos"
            )
        elif DATASET.endswith("FaceSwap"):
            DEEPFAKE_DATASET_PATH = (
                BASE_DATASET_PATH / "manipulated_sequences/FaceSwap/c23/videos"
            )
        elif DATASET.endswith("NeuralTextures"):
            DEEPFAKE_DATASET_PATH = (
                BASE_DATASET_PATH / "manipulated_sequences/NeuralTextures/c23/videos"
            )
        else:
            raise ValueError(
                f"ERROR: The specified DATASET '{DATASET}' is not recognized. Please check the dataset name."
            )
    elif DATASET.startswith("GoogleDFD"):
        BASE_DATASET_PATH = Path("/datasets/GoogleDFD/")
        REAL_DATASET_PATH = BASE_DATASET_PATH / "original_sequences/actors/c23/videos"
        DEEPFAKE_DATASET_PATH = (
            BASE_DATASET_PATH / "manipulated_sequences/DeepFakeDetection/c23/videos"
        )
    elif DATASET.startswith("CelebDFv2"):
        BASE_DATASET_PATH = Path("/datasets/celebdf2/")
        REAL_DATASET_PATH = BASE_DATASET_PATH / "Celeb-real"
        DEEPFAKE_DATASET_PATH = BASE_DATASET_PATH / "Celeb-synthesis"
    else:
        raise ValueError(
            f"ERROR: The specified DATASET '{DATASET}' is not recognized. Please check the dataset name."
        )

    return BASE_DATASET_PATH, REAL_DATASET_PATH, DEEPFAKE_DATASET_PATH


def update_paths():
    """
    Updates the global paths based on current global variables.
    """

    global OUTPUT_PATH, VIDEO_OUTPUT_DIR, SIGNAL_VISUALIZER_OUTPUT_DIR, SNR_OPTIMIZER_OUTPUT_DIR
    global SNR_ANALYZER_OUTPUT_DIR, SIGNAL_ANALYZER_OUTPUT_DIR, MASK_VISUALIZER_OUTPUT_DIR
    global PATCH_WEIGHTS_OPTIMIZER_OUTPUT_DIR, PATCH_WEIGHTS_ANALYZER_OUTPUT_DIR

    COMMON_PATH_SUFFIX = (
        Path(DATASET)
        / SUBFOLDER
        / METHOD
        / ROI_TYPE
        / f"{VIDEO_INDEX}_{VIDEO_FILE_PATH.stem}"
    )
    OUTPUT_PATH = PROJECT_PATH / 'output'
    VIDEO_OUTPUT_DIR = OUTPUT_PATH / "video_processor" / COMMON_PATH_SUFFIX.parent

    SIGNAL_VISUALIZER_OUTPUT_DIR = OUTPUT_PATH / "video_processor" / COMMON_PATH_SUFFIX

    SNR_OPTIMIZER_OUTPUT_DIR = OUTPUT_PATH / 'snr_optimizer' / COMMON_PATH_SUFFIX
    SNR_ANALYZER_OUTPUT_DIR = OUTPUT_PATH / 'snr_analyzer' / DATASET / SUBFOLDER / METHOD / ROI_TYPE

    SIGNAL_ANALYZER_OUTPUT_DIR = OUTPUT_PATH / "signal_analyzer" / COMMON_PATH_SUFFIX
    MASK_VISUALIZER_OUTPUT_DIR = OUTPUT_PATH / "mask_visualizer" / COMMON_PATH_SUFFIX

    PATCH_WEIGHTS_OPTIMIZER_OUTPUT_DIR = OUTPUT_PATH / 'patch_weights_optimizer' / VIDEO_FILE_PATH.stem
    PATCH_WEIGHTS_ANALYZER_OUTPUT_DIR = OUTPUT_PATH / 'patch_weights_analyzer'
