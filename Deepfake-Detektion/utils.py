"""
Utility functions for model training, evaluation, and resource management.

This module contains helper functions to create the PyTorch Lightning Trainer, configure
the DataModule, and test machine learning models. It also includes functions for setting 
random seeds to ensure reproducibility and for cleaning up resources after training.

Functions:
    set_seeds: Sets random seeds for reproducibility.
    cleanup_resources: Cleans up GPU and system resources.
    create_trainer: Creates a PyTorch Lightning Trainer with custom configurations.
    create_data_module: Initializes a DataModule using the provided hyperparameters.
    test_model: Trains and evaluates the model, logging results and saving final metrics.
"""

import  os
import gc
from typing import Union, Dict, Any
import warnings
import logging
from datetime import datetime
import json
import random
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    GradientAccumulationScheduler,
    StochasticWeightAveraging,
    RichProgressBar,
    RichModelSummary,
)
from rppg_facepatches.LSTM import config
from rppg_facepatches.LSTM.config import PathConfig
from rppg_facepatches.LSTM.datamodule import DataModule
from rppg_facepatches.LSTM.model import LSTMLightning

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Checkpoint directory .* exists and is not empty")
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", message="Trainer already configured with model summary callbacks")
logging.getLogger('lightning.pytorch.utilities.rank_zero').setLevel(logging.WARNING)

# Configure environment variables for CUDA and PyTorch
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TORCH_CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def set_seeds(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility of experiments.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments
    seed_everything(
        seed, workers=True
    )  # Ensures that PyTorch Lightning uses the same seed.

    torch.backends.cudnn.deterministic = True  # Ensures reproducibility with cuDNN
    torch.backends.cudnn.benchmark = False


def cleanup_resources():
    """
    Frees up resources by clearing GPU memory, synchronizing device, and running the garbage collector.

    This function is typically called after training or testing is completed to ensure that the system's
    memory and GPU resources are released.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def create_trainer(
    max_epochs: int,
    log_dir: str,
    logger=None,
    accelerator: str = 'auto',
    devices: Union[int, str] = "auto",
    early_stopping_patience: int = 5,
    swa_lrs: float = 1e-3,
    accumulation_steps: int = 4,
    gradient_clip_val: float = 0.5,
    version: str = "",
    log_every_n_steps: int = 8,
) -> Trainer:
    """
    Creates a PyTorch Lightning Trainer with a specific configuration for training and evaluating models.

    Args:
        max_epochs (int): The maximum number of epochs for training.
        log_dir (str): Directory where logs and model checkpoints will be saved.
        logger (Optional[Logger]): Custom logger (default is TensorBoard).
        accelerator (str, optional): Accelerator to use ('cpu', 'gpu', or 'auto'). Defaults to 'auto'.
        devices (Union[int, str], optional): Number of devices to use ('auto' to automatically detect). Defaults to 'auto'.
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 5.
        swa_lrs (Optional[float], optional): Learning rate for Stochastic Weight Averaging (SWA). Defaults to 1e-2.
        accumulation_steps (Optional[int], optional): Gradient accumulation steps. Defaults to 12.
        gradient_clip_val (float, optional): Gradient clipping value. Defaults to 0.5.
        version (str, optional): Experiment version for logging. Defaults to an empty string.
        log_every_n_steps (int, optional): Log every n steps. Defaults to 8.

    Returns:
        Trainer: Configured PyTorch Lightning Trainer.
    """
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Generate filename pattern for saving checkpoints
    filename_pattern = (
        f"{timestamp}_"
        f"epoch{{epoch:02d}}_"
        f"{config.MONITOR}{{{config.MONITOR}:.4e}}_"
        f"loss{{val_loss:.4e}}"
    )
    
    # Define callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=True, mode='min'),
        ModelCheckpoint(monitor='val_loss', dirpath=log_dir, filename=filename_pattern, save_top_k=1, mode='min'),
        RichProgressBar(),
        RichModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval='epoch')
    ]

    if accumulation_steps and accumulation_steps > 1:
        callbacks.append(GradientAccumulationScheduler(scheduling={0: accumulation_steps}))
    
    if swa_lrs:
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs, swa_epoch_start=int(max_epochs*0.75)))

    # Set up logger if not provided
    if logger is None:
        logger = TensorBoardLogger(save_dir=log_dir, name="trainer_logs", version=version)

    # Initialize and return Trainer
    trainer = Trainer(
        # profiler='simple',
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        logger=logger,
        default_root_dir=log_dir,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    return trainer


def create_data_module(model_config: Dict[str, Any], num_worker: int = 8) -> DataModule:
    """
    Creates and returns a DataModule for loading datasets using the specified configuration.

    Args:
        model_config (Dict[str, Any]): Model configuration parameters.
        num_worker (int, optional): Number of workers for data loading. Defaults to 8.

    Returns:
        DataModule: The initialized DataModule with batch size and balancing strategies.
    """
    return DataModule(batch_size=model_config['batch_size'],
                        balance_method=model_config['balance_method'],
                        sampler_method=model_config['sampler_method'],
                        pos_weights_bool=model_config['pos_weights_bool'],
                        num_workers=num_worker,
            )


def test_model():
    """
    Trains and evaluates a model using the specified DataModule and model configurations.

    This function initializes the DataModule with the provided configuration, loads the pre-trained model,
    and performs evaluation on the test set. Test results are logged to TensorBoard, and a summary of the
    metrics is saved to a JSON file.
    """
    # Set up logger and directory for test results
    test_log_dir = PathConfig.LOG_DIR / "testing"
    test_logger = TensorBoardLogger(
        save_dir=test_log_dir, name="Test", version=""
    )

    print("config.MODEL_CONFIG:\n", config.MODEL_CONFIG)

    # Initialize DataModule with the given configuration
    data_module = create_data_module(config.MODEL_CONFIG, num_worker=30)

    # Load pre-trained model for evaluation
    model = LSTMLightning(config.MODEL_CONFIG, data_module)
    model = torch.load(PathConfig.MODEL_DIR / f"{config.LSTM_FORWARD_METHOD}_best_model.pth")
    model.eval()

    # Create trainer for testing
    test_trainer = create_trainer(
        max_epochs=config.MODEL_CONFIG["max_epochs"],
        accelerator='auto',
        log_dir=str(test_log_dir),
        logger=test_logger,
    )

    # Start testing
    print(f"\nStarting testing for method: {config.LSTM_FORWARD_METHOD}")
    test_results = test_trainer.test(model, dataloaders=data_module)
    
    # Log the test results to TensorBoard
    test_logger.log_hyperparams({"final_testing_config": config.MODEL_CONFIG})
    for metric_name, metric_value in test_results[0].items():
        test_logger.log_metrics({metric_name: metric_value})

    # Save test results to JSON file
    results_file = PathConfig.MODEL_DIR / f"{config.LSTM_FORWARD_METHOD}_{config.ROI_TYPE}_{config.ROI_SUBTYPE}_{config.SIGNAL_TYPE}_test_results.json"
    with open(results_file, 'w', encoding='utf8') as f:
        json.dump(test_results[0], f, indent=4)

    print(f"Testing complete for method: {config.LSTM_FORWARD_METHOD}. Results saved to {results_file}.\n")
