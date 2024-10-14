"""
Configuration module for the LSTM-based model and system paths.

This module contains configurations related to the device setup, region of interest (ROI),
features, and signal processing. It also includes a `PathConfig` class to dynamically 
manage directory paths for model outputs, logs, and data, ensuring consistency across
various training and evaluation stages.

Attributes:
    DEVICE (torch.device): The computing device (CPU or CUDA).
    SEED (int): Seed for reproducibility.
    FEATURE (str): The feature type to be used for the model (e.g., 'pulse_signals').
    METHOD (str): Method used for processing signals (e.g., 'rgb', 'orgb').
    INPUT_SIZE (int): Number of input channels (1 for pulse signals, 3 for RGB signals).
    ROI_TYPE (str): Type of region of interest (e.g., 'mask').
    ROI_SUBTYPE (str): Subtype of ROI (e.g., 'full_face').
    ROI_TYPES (list): List of valid ROI types.
    MASK_ROI_SUBTYPES (list): Valid ROI subtypes for 'mask' ROI type.
    BACKGROUND_ROI_SUBTYPES (list): Valid ROI subtypes for 'background' ROI type.
    SIGNAL_TYPE (str): Type of signal to process (e.g., 'G').
    RGB_SIGNAL_TYPES (list): List of valid RGB signal combinations.
    DF (str): The dataset to use (e.g., 'GoogleDFD').
    SEQ_LENGTH (int): Sequence length for the LSTM model.
    LSTM_FORWARD_METHOD (str): The forward method to use in LSTM.
    LSTM_FORWARD_METHOD_LIST (list): List of valid forward methods based on ROI_TYPE.
    MONITOR (str): Metric to monitor during training (e.g., 'val_loss').
    MINIMIZE (bool): Whether the monitored metric should be minimized or maximized.
    MODE (str): Mode for the monitoring metric ('min' or 'max').
    DIRECTION (str): Direction for optimization ('minimize' or 'maximize').
    MODEL_CONFIG (dict): Configuration dictionary for the model, optimizer, scheduler, and dataloader.
"""

import warnings
from datetime import datetime
from pathlib import Path
import torch

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Can't initialize NVML")

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Feature and Method Configuration
FEATURE = "pulse_signals"  # Options: 'normalized_values', 'pulse_signals'
METHOD = "rgb"  # 'orgb'
INPUT_SIZE = 1 if FEATURE == "pulse_signals" else 3

# Region of Interest (ROI) Configuration
ROI_TYPE = "mask"
ROI_SUBTYPE = "full_face"
ROI_TYPES = ["patch", "mask", "background"]
MASK_ROI_SUBTYPES = ["left_cheek", "right_cheek", "cheeks", "full_face"] # "forehead", 
BACKGROUND_ROI_SUBTYPES = ["background"]

# Signal Configuration
SIGNAL_TYPE = "G"
RGB_SIGNAL_TYPES = ["R", "G", "B", "RG", "GB", "RB", "RGB", "2SR", "POS", "APOS"]

# Dataset Configuration
DF = "GoogleDFD"
SEQ_LENGTH = 128

# LSTM Configuration
LSTM_FORWARD_METHOD = "last"
LSTM_FORWARD_METHOD_LIST = (
    ["mean", "max", "mean_max", "avg_pool", "max_pool", "adaptive_pool"]
    if ROI_TYPE == "patch"
    else ["last", "mean", "max", "mean_max"]
)

# Metric Monitoring Configuration
MONITOR = "val_f1_score"  # Options: 'val_f1_score', 'val_accuracy', 'val_auroc', 'val_precision', 'val_recall', 'val_loss'
MINIMIZE = MONITOR in ["val_loss", "train_loss"]
MODE = "min" if MINIMIZE else "max"
DIRECTION = "minimize" if MINIMIZE else "maximize"

# Model and DataModule Configuration
MODEL_CONFIG = {
    # Model Configuration
    "batch_size": 80,
    "max_epochs": 500,
    "learning_rate": 0.1,
    "threshold": 0.5,
    "optimizer": "SGD",  # Options: 'SGD', 'Adam', 'RMSprop'
    "scheduler": "CosineAnnealingLR",  # Options: 'CosineAnnealingLR', 'ReduceLROnPlateau', 'ExponentialLR'
    "weight_decay": 1e-5,
    "lstm_type": "standard",  # Options: 'standard', 'multi-layer'
    "lstm_layers": 128,
    "lstm_dropout": 0.25,
    "transformer_nhead": 16,
    "transformer_dim_feedforward": 1408,
    "transformer_dropout": 0.15,
    "use_lstm_dropout": True,
    "use_lstm_layer_norm": True,
    "use_fc_layer_norm": True,
    "use_fc_batch_norm": False,
    # DataModule Configuration
    "sampler_method": "balanced_batch",  # Options: 'balanced', 'weighted', 'balanced_batch', 'weighted_batch', None
    "pos_weights_bool": True,
    "balance_method": "undersample",  # Options: 'undersample', 'oversample', 'smote', None
}


class PathConfig:
    """
    Configuration class for managing directory paths used throughout the application.
    This class consolidates all path-related settings, ensuring that file operations
    target consistent and correct locations, particularly for data output, model storage,
    and logging.

    Attributes:
        BASE_DIR (Path): The base directory for the project, determined by the location of the script.
        VIDEO_OUTPUT_DIR (Path): Directory path where video processor output is stored.
        RUN_ID (str): Unique identifier for the current execution, based on the current timestamp.
        DATA_DIR (Path): Directory path where dataset outputs are stored.
        MODEL_DIR (Path): Directory path where trained models are saved.
        LOG_DIR (Path): Directory path for storing log files.
    """

    # Define base paths and dynamic run identifier
    BASE_DIR = Path(__file__).resolve().parent
    VIDEO_OUTPUT_DIR = (
        BASE_DIR.parent / "video_processor" / "output" / "video_processor"
    )
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    DATA_DIR = BASE_DIR / "output" / ROI_TYPE / ROI_SUBTYPE
    MODEL_DIR = BASE_DIR / "models" / ROI_TYPE / ROI_SUBTYPE / SIGNAL_TYPE / RUN_ID
    LOG_DIR = BASE_DIR / "logs" / ROI_TYPE / ROI_SUBTYPE / SIGNAL_TYPE / RUN_ID

    @staticmethod
    def update_paths(roi_type: str, roi_subtype: str, signal_type: str):
        """
        Update paths dynamically based on the passed values for ROI and signal types.

        This method allows dynamic reconfiguration of data, model, and log directories
        based on the current configuration of the Region of Interest (ROI), subtype, and signal type.

        Args:
            roi_type (str): The region of interest type (e.g., 'mask', 'patch').
            roi_subtype (str): The subtype of the region of interest (e.g., 'full_face').
            signal_type (str): The signal type (e.g., 'G', 'RGB').
        """
        PathConfig.DATA_DIR = PathConfig.BASE_DIR / "output" / roi_type / roi_subtype
        PathConfig.MODEL_DIR = (
            PathConfig.BASE_DIR
            / "models"
            / roi_type
            / roi_subtype
            / signal_type
            / PathConfig.RUN_ID
        )
        PathConfig.LOG_DIR = (
            PathConfig.BASE_DIR
            / "logs"
            / roi_type
            / roi_subtype
            / signal_type
            / PathConfig.RUN_ID
        )

    @staticmethod
    def init_directories():
        """
        Creates the necessary directories for data, models, and logs if they do not already exist.

        This method ensures that all directories required for model training, saving,
        and logging are properly set up at the beginning of the process.
        """
        Path(PathConfig.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(PathConfig.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        Path(PathConfig.LOG_DIR).mkdir(parents=True, exist_ok=True)
        # Path(PathConfig.MODEL_STATE_DICT_DIR).mkdir(parents=True, exist_ok=True)
