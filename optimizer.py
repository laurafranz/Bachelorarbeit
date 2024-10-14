"""
Hyperparameter Optimization Module for LSTM Model using Optuna and PyTorch Lightning.

This module provides functionality for hyperparameter optimization of LSTM-based models, specifically designed for sequence-based tasks such as time series classification and sequence learning. The optimization process leverages the Optuna library and integrates with PyTorch Lightning for streamlined model training, evaluation, and logging.

Classes:
    - `ExceptionHandler`: Manages and logs exceptions that occur during the hyperparameter optimization process.
    - `ResourceManager`: Ensures proper cleanup of system resources (e.g., GPU memory) between trials.
    - `Objective`: Defines the optimization objective, including the hyperparameter search space, model training, and evaluation.
    - `HyperparameterOptimizer`: Manages the optimization process over multiple iterations, logs results, and generates visualizations.
"""

import warnings
import os
import logging
from logging import Logger
import time
from itertools import combinations
from collections import defaultdict
from typing import Dict, Any, Union, List, Optional, Tuple
import io
from pathlib import Path
import json
import gc
import shutil
import psutil
from PIL import Image
import numpy as np
import plotly.io as pio
import optuna
from optuna import Trial, Study
import optuna.visualization as ov
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from  pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import torch
from tabulate import tabulate
from termcolor import colored

from rppg_facepatches.LSTM.config import PathConfig
from rppg_facepatches.LSTM import config
from rppg_facepatches.LSTM.model import LSTMLightning
from rppg_facepatches.LSTM.utils import create_trainer, create_data_module
from rppg_facepatches.LSTM.datamodule import DataModule

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Param .* unique value length is less than 2.")
warnings.filterwarnings("ignore", category=UserWarning, message="Can't initialize NVML")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TORCH_CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Setzen Sie eine Grenze für die CUDA-Speichernutzung
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)


class ExceptionHandler:
    """
    A class responsible for handling exceptions that occur during Optuna trials.

    This class provides robust exception handling for Optuna-based hyperparameter optimization processes.
    It catches common errors such as CUDA out-of-memory, disk space errors, and general runtime errors. 
    It logs error details and system states like CPU, memory, and disk usage, provides suggestions for resolving 
    the errors, and cleans up resources when necessary. 

    Attributes:
        logger (Logger): Logger instance for recording error messages, system state, and recommendations.
    """

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def handle_exception(self, e: Exception, context_info: Optional[Union[Trial, int]]) -> None:
        """
        Handles exceptions that occur during the trial.

        Depending on the exception type, this method logs the error, suggests next steps, and performs actions like 
        freeing up GPU memory or cleaning up disk space.

        Args:
            e (Exception): The exception that occurred.
            context_info (Union[Trial, int], optional): Information about the current Optuna trial or iteration number.
        """
        if isinstance(context_info, Trial):
            context = f"Trial {context_info.number}"
        elif isinstance(context_info, int):
            context = f"Iteration {context_info}"
        else:
            context = "Unknown context"
        
        # Handle specific types of exceptions
        if isinstance(e, RuntimeError):
            if "CUDA out of memory" in str(e):
                self._handle_cuda_oom(context_info)
            elif "No space left on device" in str(e):
                self._handle_disk_space_error(context_info)
            else:
                self.logger.error("%s failed with RuntimeError: %s", context, str(e))
                self._log_system_state()
        elif isinstance(e, torch.cuda.CudaError):
            self._handle_cuda_error(context_info)
        elif isinstance(e, ValueError):
            self.logger.error("%s failed due to invalid value: %s", context, str(e))
            if isinstance(context_info, Trial):
                self._log_parameter_state(context_info)
        elif isinstance(e, MisconfigurationException):
            self.logger.error("%s failed due to misconfiguration: %s", context, str(e))
            if isinstance(context_info, Trial):
                self._log_parameter_state(context_info)
        else:
            # Log unexpected errors and CUDA memory summary if possible
            self.logger.error("%s failed with unexpected error: %s: %s", context, type(e).__name__, str(e), exc_info=True)
            if torch.cuda.is_available():
                try:
                    memory_summary = torch.cuda.memory_summary()
                    self.logger.error("CUDA memory summary:\n%s", memory_summary)
                except Exception as mem_e:
                    self.logger.error("Failed to retrieve CUDA memory summary: %s: %s", type(mem_e).__name__, str(mem_e))
            self._log_system_state()

        self._suggest_next_steps(e)

    def _handle_cuda_oom(self, context: str):
        """
        Handles CUDA Out of Memory (OOM) errors by freeing up GPU memory and logging suggestions.

        Args:
            context (str): Contextual information about the error (e.g., trial number or iteration).
        """
        self.logger.error("%s failed due to CUDA out of memory", context)
        self._log_gpu_state()
        torch.cuda.empty_cache()
        self.logger.info("GPU cache cleared. Consider reducing model size or batch size.")

    def _handle_disk_space_error(self, context: str):
        """
        Handles disk space errors when the device runs out of storage by attempting to free up space.

        Args:
            context (str): Contextual information about the error (e.g., trial number or iteration).
        """
        self.logger.error("%s failed due to insufficient disk space", context)
        if self._clean_up_space():
            self.logger.info("Space cleaned up. You may retry the operation.")
        else:
            self.logger.warning("Could not free up enough space. Manual intervention may be required.")
        self._log_disk_state()

    def _handle_cuda_error(self, context: str):
        """
        Handles generic CUDA errors by cleaning up resources and logging the GPU state.

        Args:
            context (str): Contextual information about the error (e.g., trial number or iteration).
        """
        self.logger.error("CUDA error occurred in %s. Attempting to clean up", context)
        self._log_gpu_state()
        if not self._clean_up_space():
            self.logger.critical("Failed to clean up enough space. Consider reducing model complexity or dataset size.")

    def _log_system_state(self):
        """
        Logs the current system state including CPU, memory, and disk usage.
        """
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage("/").percent
        self.logger.info("System State - CPU: %d%%, Memory: %d%%, Disk: %d%%", cpu_percent, memory_percent, disk_percent)

    def _log_gpu_state(self):
        """
        Logs the current GPU memory usage if CUDA is available.
        """
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.logger.info("GPU %d Memory - Allocated: %.2fGB, Cached: %.2fGB", 
                                 i, 
                                 torch.cuda.memory_allocated(i) / 1e9, 
                                 torch.cuda.memory_reserved(i) / 1e9)

    def _log_disk_state(self):
        """
        Logs the current disk usage.
        """
        total, used, free = shutil.disk_usage("/")
        self.logger.info("Disk Usage - Total: %dGB, Used: %dGB, Free: %dGB", 
                         total // (2**30), used // (2**30), free // (2**30))

    def _log_parameter_state(self, trial: Trial):
        """
        Logs the current parameters of the trial.

        Args:
            trial (Trial): The current Optuna trial.
        """
        self.logger.info("Current trial parameters: %s", str(trial.params))

    def _suggest_next_steps(self, e: Exception):
        """
        Suggests next steps based on the type of exception encountered.

        Args:
            e (Exception): The exception that occurred.
        """
        if "CUDA out of memory" in str(e):
            self.logger.info("Suggested next steps: Reduce batch size, decrease model complexity, or use gradient accumulation.")
        elif "No space left on device" in str(e):
            self.logger.info("Suggested next steps: Free up disk space, use a larger disk, or reduce logging/checkpointing frequency.")
        elif isinstance(e, ValueError):
            self.logger.info("Suggested next steps: Check your hyperparameter ranges and ensure all inputs are valid.")
        elif isinstance(e, MisconfigurationException):
            self.logger.info("Suggested next steps: Review your model and trainer configurations for inconsistencies.")
        else:
            self.logger.info("Suggested next steps: Review the error message and stack trace, and consider simplifying your experiment.")

    def _clean_up_space(self, days_old: int = 7, min_space_gb: int = 10):
        """
        Cleans up disk space by removing old log files and cached data.

        Args:
            days_old (int): Remove files older than this many days.
            min_space_gb (int): Minimum space in GB to try to free up.

        Returns:
            bool: True if enough space was freed, False otherwise.
        """
        self.logger.info("Attempting to clean up disk space.")
        
        # Directories to clean
        dirs_to_clean = [
            Path(os.environ.get("TORCH_HOME", "~/.cache/torch")),  # PyTorch cache
            Path("/tmp"),  # Temporary files
        ]

        bytes_removed = 0
        min_space_bytes = min_space_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        current_time = time.time()

        # Clean up files older than specified days
        for directory in dirs_to_clean:
            if not directory.exists():
                continue
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    file_path = Path(root) / name
                    try:
                        if current_time - file_path.stat().st_mtime > days_old * 86400:  # 86400 seconds in a day
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            bytes_removed += file_size
                            self.logger.info(f"Removed: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error while removing {file_path}: {e}")
                for name in dirs:
                    dir_path = Path(root) / name
                    try:
                        if not any(dir_path.iterdir()):  # Check if directory is empty
                            dir_path.rmdir()
                            self.logger.info(f"Removed empty directory: {dir_path}")
                    except Exception as e:
                        self.logger.error(f"Error while removing directory {dir_path}: {e}")
            if bytes_removed >= min_space_bytes:
                break
        space_freed_gb = bytes_removed / (1024 * 1024 * 1024)
        self.logger.info(f"Total space freed: {space_freed_gb:.2f} GB")

        # Check if we"ve freed up enough space
        if bytes_removed >= min_space_bytes:
            self.logger.info("Successfully freed up requested disk space.")
            return True
        else:
            self.logger.warning("Could not free up requested disk space.")
            return False


class ResourceManager:
    """
    Manages system resources such as memory and GPU during Optuna trials.

    This class is responsible for freeing up memory and clearing GPU caches after each Optuna trial or at specific 
    points during the model"s lifecycle. It ensures proper resource management, which helps prevent memory-related 
    issues like out-of-memory errors during hyperparameter optimization.

    Attributes:
        logger (Logger): Logger instance for recording events related to resource management.
    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def cleanup(self, trial: Optional[Trial] = None, data_module: Optional[DataModule] = None, model: Optional[LSTMLightning] = None):
        """
        Frees up system resources like GPU memory, clears data module and model states, and ensures garbage collection.

        This method is responsible for cleaning up the resources after each trial or at certain points in the model"s 
        lifecycle to ensure efficient memory usage. It clears the data module, model states, and GPU caches, and triggers 
        Python"s garbage collector to reclaim unused memory.

        Args:
            trial (Optional[Trial]): The current Optuna trial, if available. Logs the trial number after cleanup.
            data_module (Optional[DataModule]): The data module that needs to be cleaned up.
            model (Optional[LSTMLightning]): The model to clean up.
        """
        # Clean up data module if provided
        if data_module:
            data_module.cleanup()
        
        # Clean up model if provided
        if model:
            model.cleanup()
        
        # Trigger Python garbage collection to free up memory
        gc.collect()

        # If CUDA is available, synchronize GPU and clean up memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Log cleanup completion based on whether a trial is provided
        if trial is not None:
            self.logger.info("Cleanup completed for trial %d", trial.number)
        else:
            self.logger.info("Cleanup completed.")


class Objective:
    """
    Objective class for hyperparameter optimization using Optuna.

    This class defines the search space for hyperparameters, manages the training and evaluation of the model,
    handles exceptions, and logs results. It is designed to work with the PyTorch Lightning framework, specifically 
    for models like LSTMLightning that are used for sequence-based tasks.

    Attributes:
        num_classes (int): The number of output classes for the model.
        fixed_params (Dict[str, Any]): A dictionary of fixed hyperparameters that do not change during the optimization.
        iteration_log_dir (Path): The directory where logs for each iteration will be stored.
        best_model (Optional[LSTMLightning]): The best model obtained during the optimization process.
        best_metric_value (float): The best metric value obtained during the optimization process, initialized based on the optimization direction (minimize/maximize).
        logger (Logger): Logger instance for recording events, warnings, and errors.
        all_trials (list): A list of all trials executed during the optimization process.
        data_module (Optional[DataModule]): The data module used during the optimization process, initialized in the `__call__` method.
    """

    def __init__(self, num_classes: int, fixed_params: Dict[str, Any], iteration: int, iteration_log_dir: str):
        self.num_classes: int = num_classes
        self.fixed_params: Dict[str, Any] = fixed_params
        self.iteration = iteration
        self.iteration_log_dir: Path = Path(iteration_log_dir)
        self.best_model: Optional[LSTMLightning] = None
        self.best_metric_value: float = float("inf") if config.MINIMIZE else float("-inf")
        self.best_loss_value: float = float('inf')
        self.logger: Logger = logging.getLogger(__name__)
        self.all_trials: list = []
        self.data_module: Optional[DataModule] = None
        self.model: Optional[LSTMLightning] = None
        
        self.exception_handler = ExceptionHandler(self.logger)
        self.resource_manager = ResourceManager(self.logger)
        
    def __call__(self, trial: Trial):
        """
        Executes a single trial of hyperparameter optimization.

        This method sets up the hyperparameters, data module, and model, then trains and evaluates the model.
        It handles exceptions that may occur during the trial, logging relevant information and returning
        the best metric value obtained.

        Args:
            trial (Trial): An Optuna trial object that represents a single run of hyperparameter optimization.

        Returns:
            float: The best metric value obtained during the trial.
        """

        hparams = self._suggest_hyperparameters(trial)
        trial_log_dir, trial_logger = self._setup_trial_logging(trial)

        try:
            # Create the data module and model
            self.data_module = create_data_module(hparams, num_worker=30) # , num_worker=4
            self.model = LSTMLightning(hparams, self.data_module)

            # Create the trainer
            trainer = create_trainer(max_epochs=hparams["max_epochs"], log_dir=trial_log_dir, logger=trial_logger)

            self.logger.info("Hyperparameters for trial %d: %s", trial.number, hparams)

            # Tune the learning rate
            self._tune_learning_rate(trainer, trial, trial_logger, self.iteration, hparams)

            # Log the hyperparameters in the terminal before any tuning
            self.logger.info("Hyperparameters for trial %d: %s", trial.number, hparams)

            # Train and evaluate the model
            best_metric_value = self._train_and_evaluate(trainer, trial, trial_logger)

            # Log and store the trial results
            self.all_trials.append(trial)
            self.logger.info("Trial %d completed with best metric value: %4e\n", trial.number, self.best_metric_value)
            
            return best_metric_value
        except optuna.exceptions.TrialPruned as e:
            self.logger.info("Trial %d pruned (Iteration %d) %s\n", trial.number, self.iteration, e)
            raise e
        except Exception as e:
            self.exception_handler.handle_exception(e, trial)
            return float("inf") if config.MINIMIZE else float("-inf")

    def _tune_learning_rate(self, trainer: Trainer, trial: Trial, logger: TensorBoardLogger, iteration: int, hparams: Dict[str, Any], max_steps: int = 100):
        """
        Tunes the learning rate for the model using the PyTorch Lightning Tuner.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            iteration_logger (TensorBoardLogger): The logger to use for tracking the tuning process.
            iteration (int): The current iteration number.
            hparams (Dict[str, Any]): The hyperparameters dictionary to update with the new learning rate.
            max_steps (int, optional): The maximum number of steps for the learning rate finder. Defaults to 150.
        """

        try:
            # hparams["learning_rate"] = 0.1
            
            tuner = Tuner(trainer)
            
            # Use the tuner to find the optimal learning rate
            lr_finder = tuner.lr_find(self.model, self.data_module, num_training=max_steps)
            
            # # Auto-scale batch size by growing it exponentially (default)
            # batch_size = tuner.scale_batch_size(self.model, mode="power", datamodule=self.data_module)
            # print(f"Batch size (power scaling): {batch_size}") # 32768, 16384
            # # Auto-scale batch size with binary search
            # batch_size = tuner.scale_batch_size(self.model, mode="binsearch", datamodule=self.data_module)
            # print(f"Batch size (binary search): {batch_size}") # 32768, 16384

            # # Plot lr
            # fig = lr_finder.plot(suggest=True)
            # fig.show()
            
            # Update the learning rate in the hyperparameters
            new_lr = lr_finder.suggestion()
            hparams["learning_rate"] = new_lr

            # Log the learning rate finder plot
            figure_name = f"LR Finder - Iteration {iteration}, Trial {trial.number}, LSTM Forward Method {config.LSTM_FORWARD_METHOD}"
            logger.experiment.add_figure(figure_name, lr_finder.plot(suggest=True), global_step=trial.number)
        except Exception as e:
            self.logger.error("Failed to tune the learning rate: %s", e)
            raise e

    def get_best_trials(self, n: int = 3):
        """
        Retrieves the top `n` best trials based on the monitored metric.

        Args:
            n (int): The number of top trials to return. Defaults to 3.

        Returns:
            list: A list of the best trials, sorted by the monitored metric in descending order.
        """

        return sorted(self.all_trials, key=lambda t: t.user_attrs.get(config.MONITOR, float("-inf")), reverse=True)[:n]

    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggests a set of hyperparameters for a given trial.

        This method defines the search space for various hyperparameters, including LSTM configuration,
        optimizer settings, dropout, and learning rate scheduler.

        Args:
            trial (Trial): An Optuna trial object that represents a single run of hyperparameter optimization.

        Returns:
            Dict[str, Any]: A dictionary containing the suggested hyperparameters.
        """

        # Start by copying the fixed params
        hparams = self.fixed_params.copy()
        self.logger.debug("Fixed parameters: %s", hparams)

        # Add dynamically suggested hyperparameters
        try:
            hparams.update({
                "lstm_type": trial.suggest_categorical("lstm_type", ["multi_layer", "standard"]), # Options: standard, multi_layer
                "use_lstm_dropout": trial.suggest_categorical("use_lstm_dropout", [True]),
                "use_lstm_layer_norm": trial.suggest_categorical("use_lstm_layer_norm", [True, False]),
                "use_fc_batch_norm": trial.suggest_categorical("use_fc_batch_norm", [False]),
                "use_fc_layer_norm": trial.suggest_categorical("use_fc_layer_norm", [True, False]),
                "balance_method": trial.suggest_categorical("balance_method", ["undersample", "smote"]), # Options: smote, oversample, undersample, None
                "sampler_method": trial.suggest_categorical("sampler_method", ["balanced_batch"]), # Options: balanced, weighted, balanced_batch, weighted_batch, None
                "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]), # Options: SGD, RMSprop, Adam, AdamW
                "scheduler": trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "ReduceLROnPlateau"]), # Options: ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
                "weight_decay": trial.suggest_categorical("weight_decay", [1e-6, 1e-5]),
                "batch_size": trial.suggest_int("batch_size", 80, 144, step=8), # 104-184
                "threshold": trial.suggest_float("threshold", 0.45, 0.55), # 0.35-0.45
            })
            
            # Conditionally set pos_weights_bool based on the value of balance_method
            if hparams["balance_method"] is None:
                hparams["pos_weights_bool"] = True
            else:
                hparams["pos_weights_bool"] = trial.suggest_categorical("pos_weights_bool", [True, False])

        except Exception as e:
            self.logger.error("Error while suggesting hyperparameters: %s", e)
            raise

        self.logger.debug("Hyperparameters after basic suggestions: %s", hparams)

        # Suggest optimizer and scheduler configurations
        try:
            hparams.update(self._suggest_optimizer_config(trial, hparams["optimizer"]))
            hparams.update(self._suggest_scheduler_config(trial, hparams["scheduler"]))
        except Exception as e:
            self.logger.error("Error in optimizer or scheduler configuration: %s", e)
            raise
        
        self.logger.debug("Hyperparameters after optimizer and scheduler: %s", hparams)

        # Suggest LSTM layer configuration
        try:
            lstm_config = self._suggest_lstm_layer_config(trial, hparams["lstm_type"])
            hparams.update(lstm_config)
        except Exception as e:
            self.logger.error("Error in LSTM layer configuration: %s", e)
            raise

        self.logger.debug("Hyperparameters after LSTM config: %s", hparams)

        # Conditionally suggest dropout configuration based on the number of LSTM layers
        try:
            lstm_layers = lstm_config["lstm_layers"]
            if (isinstance(lstm_layers, int) and lstm_layers > 1) or (isinstance(lstm_layers, list) and len(lstm_layers) > 1):
                hparams.update(self._suggest_dropout_config(trial, True))
            else:
                hparams.update(self._suggest_dropout_config(trial, False))
        except Exception as e:
            self.logger.error("Error in dropout configuration: %s", e)
            raise

        self.logger.debug("Hyperparameters after dropout config: %s", hparams)

        if config.ROI_TYPE == "patch":
            try:
                transformer_config = self._suggest_transformer_config(trial, lstm_layers)
                hparams.update(transformer_config)
            except Exception as e:
                self.logger.error("Error in transformer configuration: %s", e)
                raise
        
        self.logger.debug("Final hyperparameters: %s", hparams)

        trial.set_user_attr("hparams", hparams)

        return hparams

    def _suggest_transformer_config(self, trial: Trial, embed_dim) -> Dict[str, Any]:
        """
        Suggests the configuration for a Transformer model.

        Args:
            trial (Trial): An Optuna trial object that represents a single run of hyperparameter optimization.

        Returns:
            Dict[str, Any]: A dictionary containing the Transformer configuration.
        """
        
        if isinstance(embed_dim, list):
            embed_dim = embed_dim[-1]
        elif isinstance(embed_dim, int):
            embed_dim = embed_dim
        else:
            self.logger.error("Unexpected embed_dim type: %s", type(embed_dim))

        # Ensure nhead divides d_model evenly
        valid_nhead_options = [n for n in range(4, 17, 4) if embed_dim % n == 0]
        self.logger.debug("Valid nhead options: %s", valid_nhead_options)

        if not valid_nhead_options:
            error_message = f"Could not find a valid nhead that divides embed_dim ({embed_dim}) evenly."
            self.logger.error(error_message)
            raise ValueError(error_message)
        
        # Suggest hyperparameters for the Transformer
        transformer_config = {
            "transformer_nhead": trial.suggest_categorical("transformer_nhead", [8, 16]),  # 16 # Choose a valid nhead
            "transformer_num_layers": trial.suggest_categorical("transformer_num_layers", [3]),  # Number of transformer layers
            "transformer_dim_feedforward": trial.suggest_categorical("transformer_dim_feedforward", [1408]),  # Size of the feedforward network
            "transformer_dropout": trial.suggest_float("transformer_dropout", 0.1, 0.17)  # Dropout rate
        }

        return transformer_config

    def _suggest_lstm_layer_config(self, trial: Trial, lstm_type: str) -> Dict[str, Union[int, List[int]]]:
        """
        Suggests the configuration for LSTM layers based on the type of LSTM.

        Args:
            trial (Trial): An Optuna trial object that represents a single run of hyperparameter optimization.
            lstm_type (str): The type of LSTM layer ("standard" or "multi_layer").

        Returns:
            Dict[str, Union[int, List[int]]]: A dictionary containing the configuration for LSTM layers.
        """

        if lstm_type == "standard":
            lstm_size = trial.suggest_int("lstm_layers", 128, 384, step=128) # 224
            return {"lstm_layers": lstm_size}
        
        elif lstm_type == "multi_layer":
            lstm_layers = trial.suggest_int("lstm_num_layers", 2, 3)
            lstm_sizes = [trial.suggest_int(f"layer_sizes_{i}", 128, 384, step=16) for i in range(lstm_layers)]

            # lstm_layers = trial.suggest_int("lstm_num_layers", 2, 3)  # Number of layers for multi-layer LSTM

            # # Define the size for the first layer
            # lstm_sizes = []
            # previous_size = trial.suggest_int("previous_size", 256, 512, step=16)
            # lstm_sizes.append(previous_size)

            # # Suggest sizes for the subsequent layers, ensuring that each is smaller than the previous one
            # for i in range(1, lstm_layers):
            #     upper_limit = lstm_sizes[-1]  # Ensure next layer size is smaller
            #     if upper_limit < 128:
            #         raise ValueError("Invalid configuration: next layer size cannot be smaller than 128.")
            #     next_layer_size = trial.suggest_int(f"layer_sizes_{i}", 128, upper_limit, step=16)
            #     lstm_sizes.append(next_layer_size)

            return {"lstm_layers": lstm_sizes}
        else:
            raise ValueError(f"Unknown lstm type: {lstm_type}")
        
    def _suggest_optimizer_config(self, trial: Trial, optimizer_name: str) -> Dict[str, float]:
        """
        Suggests the configuration for the optimizer, particularly the momentum parameter.

        Args:
            trial (Trial): An Optuna trial object that represents a single run of hyperparameter optimization.
            optimizer_name (str): The name of the optimizer being used ("SGD", "Adam", "RMSprop").

        Returns:
            Dict[str, float]: A dictionary containing the optimizer configuration.
        """

        return {"momentum": trial.suggest_float("momentum", 0.5, 0.99) if optimizer_name not in ["Adam", "AdamW"] else 0.0}

    def _suggest_scheduler_config(self, trial: Trial, scheduler: str) -> Dict[str, Any]:
        """
        Suggests the configuration for the learning rate scheduler.

        Args:
            trial (Trial): An Optuna trial object that represents a single run of hyperparameter optimization.
            scheduler (str): The name of the learning rate scheduler ("CosineAnnealingLR", "ReduceLROnPlateau", "ExponentialLR").

        Returns:
            Dict[str, Any]: A dictionary containing the scheduler configuration.
        """

        if scheduler == "CosineAnnealingLR":
            warmup_percentage = trial.suggest_float("warmup_percentage", 0.30, 0.40)
            return {"cosine_t_max": int(warmup_percentage * self.fixed_params["max_epochs"])}
        elif scheduler == "ReduceLROnPlateau":
            return {"factor": trial.suggest_float("factor", 0.1, 0.3, log=True), "patience": trial.suggest_int("patience", 8, 12)}
        elif scheduler == "ExponentialLR":
            return {"gamma": trial.suggest_float("gamma", 0.95, 0.99, log=True)}
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    def _suggest_dropout_config(self, trial: Trial, use_droput: bool) -> Dict[str, float]:
        """
        Suggests the dropout configuration based on whether dropout is enabled.

        Args:
            trial (Trial): An Optuna trial object that represents a single run of hyperparameter optimization.
            use_lstm_dropout (bool): A flag indicating whether dropout is enabled.

        Returns:
            Dict[str, float]: A dictionary containing the dropout configuration.
        """

        if use_droput is True:
            dropout = trial.suggest_float("lstm_dropout", 0.20, 0.29) # 0.15-0.25
            return {"lstm_dropout": dropout}
        else:
            return {"lstm_dropout": 0.0}
        
    def _setup_trial_logging(self, trial: Trial) -> Tuple[Path, TensorBoardLogger]:
        """
        Sets up the logging directory and logger for a given trial.

        Args:
            trial (Trial): The current Optuna trial.

        Returns:
            Tuple[Path, TensorBoardLogger]: The directory path for trial logs and the TensorBoard logger.
        """

        trial_name = f"trial{trial.number + 1}"
        trial_log_dir = self.iteration_log_dir / trial_name
        trial_logger = TensorBoardLogger(save_dir=PathConfig.LOG_DIR / self.iteration_log_dir, name=trial_name, version="")
        return trial_log_dir, trial_logger

    def _train_and_evaluate(self, trainer: Trainer, trial: Trial, trial_logger: TensorBoardLogger):
        """
        Trains and evaluates the model within a trial.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            trial (Trial): The current Optuna trial.
            trial_logger (TensorBoardLogger): The logger for the trial.
        """

        self.logger.info("Starting training for trial %d", trial.number)
        
        # Log hyperparameters for the trial
        trial_logger.log_hyperparams(trial.params)
        
        # Start training
        trainer.fit(self.model, datamodule=self.data_module)
        
        # Collect validation metrics
        val_metrics = {
            f"Validation/Monitor_{config.MONITOR}": trainer.callback_metrics.get(config.MONITOR),
            "Validation/Accuracy": trainer.callback_metrics.get("val_accuracy"),
            "Validation/Precision": trainer.callback_metrics.get("val_precision"),
            "Validation/Recall": trainer.callback_metrics.get("val_recall"),
            "Validation/F1 Score": trainer.callback_metrics.get("val_f1_score"),
            "Validation/AUROC": trainer.callback_metrics.get("val_auroc"),
            "Validation/Loss": trainer.callback_metrics.get("val_loss"),
        }
        
        # Log validation metrics to TensorBoardLogger
        trial_logger.log_metrics(val_metrics, step=trainer.current_epoch)
        
        # Log validation metrics to the terminal
        for metric_name, metric_value in val_metrics.items():
            if metric_value is not None:
                self.logger.info("%s: %.4g", metric_name, metric_value)

        # Set user attributes for the trial
        trial.set_user_attr("iteration", self.iteration)
        for key, value in val_metrics.items():
            trial.set_user_attr(key.split("/")[-1].lower(), value)
        
        # Handle invalid validation metrics
        # val_metric = val_metrics[f"Validation/Monitor_{config.MONITOR}"]
        val_metric = val_metrics.get(f"Validation/Monitor_{config.MONITOR}")

        if val_metric is None or not np.isfinite(val_metric):
            self.logger.warning("Invalid val_metric: %s", str(val_metric))
            return float("inf") if config.MINIMIZE else float("-inf")
        
        # Report the metric to Optuna and check if it"s a new best
        trial.report(val_metric, trainer.current_epoch)

        # Check if val_metric meet the criteria for new best
        is_better = (val_metric < self.best_metric_value if config.MINIMIZE else val_metric > self.best_metric_value)
                    
        # Update the best model if val metric is better
        if is_better:
            self.best_loss_value = val_metrics.get("Validation/Loss")
            self.best_metric_value = val_metric
            self.best_model = self.model
            self.logger.info("New best model found in trial %d (Iteration %d) with %s: %4e",
                        trial.number, self.iteration, config.MONITOR, self.best_metric_value)
        
        # Store the best metric and model for the trial
        trial.set_user_attr("best_loss_value", self.best_loss_value)
        trial.set_user_attr("best_metric_value", self.best_metric_value)
        trial.set_user_attr("best_model", self.best_model)
        
        # Handle trial pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Ensure synchronization if using GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.logger.info("Evaluation completed for trial %d (Iteration %d)", trial.number, self.iteration)

        return self.best_metric_value


class HyperparameterOptimizer:
    """
    A class to manage the hyperparameter optimization process using Optuna and PyTorch Lightning.

    This class orchestrates the entire optimization process including setting up the data module,
    creating and tuning the model, logging results, and generating visualizations for analysis.
    
    Attributes:
        best_metric (float): The best metric value achieved during the optimization process.
        best_model_config (Dict[str, Any]): The configuration of the best model found.
        best_model (LSTMLightning): The best model found during the optimization process.
        best_trials (List[Trial]): A list of the best trials from the optimization process.
        n_iterations (int): The number of iterations to run for optimization.
        method_log_dir (Path): The directory to save logs and outputs of the optimization process.
        data_module_class (Type[DataModule]): The data module class used for data handling.
        model_class (Type[LSTMLightning]): The model class used for training and evaluation.
        objective (Optional[Objective]): The objective function that defines the optimization problem.
        balance_method (str): The method used for balancing the dataset.
    """

    def __init__(self, n_iterations: int, method_log_dir: str):
        self.best_metric: float = float("-inf")
        self.best_model_config: Dict[str, Any] = config.MODEL_CONFIG.copy()
        self.best_model: LSTMLightning = None
        self.best_trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None

        self.n_iterations = n_iterations
        self.method_log_dir: Path = method_log_dir
        self.data_module: Optional[DataModule] = None
        self.model: Optional[LSTMLightning] = None
        self.objective: Optional[Objective] = None
        self.balance_method: str = self.best_model_config["balance_method"]

        self.logger = logging.getLogger(__name__)
        self.exception_handler = ExceptionHandler(self.logger)
        self.resource_manager = ResourceManager(self.logger)

    def optimize(self) -> Tuple[float, Dict[str, Any]]:
        """
        Runs the hyperparameter optimization process over a specified number of iterations.

        This method performs the optimization for each iteration, tunes the learning rate,
        processes the results, and logs the best trials and their metrics.

        Returns:
            Tuple[float, Dict[str, Any]]: The best metric value and the configuration of the best model.
        """

        for iteration in range(self.n_iterations):
            iteration_log_dir, iteration_logger = self._setup_iteration_logging(iteration)
            self.logger.info("Starting iteration %d of %d", iteration + 1, self.n_iterations)

            try:
                self.objective = Objective(num_classes=2, fixed_params=self.best_model_config, iteration=iteration+1, iteration_log_dir=iteration_log_dir)
                study = optuna.create_study(direction=config.DIRECTION)
                study.optimize(self.objective, n_trials=10, gc_after_trial=True)
                self.process_optimization_results(study, iteration_logger, iteration)
                iteration_logger.log_metrics({f"best_{config.MONITOR}": self.best_trial.user_attrs["best_metric_value"]})
            except Exception as e:
                self.exception_handler.handle_exception(e, iteration+1)
        
        # Save the best model and config for the current method
        self._save_best_model_and_config()
        
        # Debugging: Print before updating config.MODEL_CONFIG
        print("\nBefore updating config.MODEL_CONFIG:")
        print("Best model config:\n", self.best_model_config)
        print("Current config.MODEL_CONFIG:\n", config.MODEL_CONFIG)

        # Update the global configuration with the best model config
        config.MODEL_CONFIG = self.best_model_config
        
        # Debugging: Print after updating config.MODEL_CONFIG
        print("\nAfter updating config.MODEL_CONFIG:")
        print("Updated config.MODEL_CONFIG:\n", config.MODEL_CONFIG)
        # config.MODEL_CONFIG.update(self.best_model_config)

        # return self.best_metric, self.best_model_config
    
    def _save_best_model_and_config(self):
        """
        Saves the best model and its configuration to disk.
        """
        
        # Paths for saving the best model and configuration
        model_save_path = PathConfig.MODEL_DIR / f"{config.LSTM_FORWARD_METHOD}_best_model.pth"
        config_save_path = PathConfig.MODEL_DIR / f"{config.LSTM_FORWARD_METHOD}_best_model_config.json"
        metrics_save_path = PathConfig.MODEL_DIR / f"{config.LSTM_FORWARD_METHOD}_best_metrics.json"

        # Save the best model
        torch.save(self.best_model, model_save_path)

        # Save the best model configuration
        with open(config_save_path, "w", encoding="utf8") as config_file:
            json.dump(self.best_model_config, config_file, indent=4)

        # Convert best metric to a float (or another serializable type) if it"s a tensor
        best_metric_value = self.best_metric.item() if isinstance(self.best_metric, torch.Tensor) else self.best_metric

        # Save the best validation metric and value
        best_metrics = {
            "best_metric_name": config.MONITOR,
            "best_metric_value": best_metric_value
        }
        with open(metrics_save_path, "w", encoding="utf8") as metrics_file:
            json.dump(best_metrics, metrics_file, indent=4)

        # Logging the save operation
        self.logger.info("Saved best model, config, and metrics for method \"%s\".\n", config.LSTM_FORWARD_METHOD)

    def _setup_iteration_logging(self, iteration: int) -> Tuple[Path, TensorBoardLogger]:
        """
        Sets up logging for a specific iteration of the optimization process.

        Args:
            iteration (int): The current iteration number.

        Returns:
            Tuple[Path, TensorBoardLogger]: The path to the log directory and the logger for this iteration.
        """

        iteration_name = f"iteration_{iteration + 1}"
        iteration_log_dir = self.method_log_dir / iteration_name
        iteration_logger = TensorBoardLogger(save_dir=iteration_log_dir, name=iteration_name, version="")
        return iteration_log_dir, iteration_logger

    def process_optimization_results(self, study: Study, iteration_logger: TensorBoardLogger, iteration: int):
        """
        Processes the results of the hyperparameter optimization, logs the best trials, and updates the best model configuration.

        Args:
            study (Study): The Optuna study object containing the results of the optimization.
            iteration_logger (TensorBoardLogger): The logger to use for tracking the optimization results.
            iteration (int): The current iteration number.
        """

        self.logger.info("Number of finished trials: %d", len(study.trials))

        best_trials = self.objective.get_best_trials()
        for trial in best_trials:
            trial.iteration = iteration
        self.best_trials.extend(best_trials)

        # Update the best trial and log its metrics and configuration
        self._update_and_log_best_trial(study, iteration)

        self.logger.info("Top Trials:")
        for i, trial in enumerate(best_trials):
            self.logger.info("%d. Iteration %d Trial %d - Value for %s: %s",
                             i + 1, trial.iteration + 1, trial.number + 1, config.MONITOR, trial.user_attrs.get("best_metric_value", "N/A"))
        
        self.best_trial = self._get_best_trial(study)

        best_trial_value = self.best_trial.user_attrs["best_metric_value"]
        best_model_config = self.best_trial.user_attrs["hparams"]
        best_model = self.best_trial.user_attrs["best_model"]
        trial_iteration = self.best_trial.user_attrs["iteration"]

        self.best_metric = best_trial_value
        self.best_model_config = best_model_config
        self.best_model = best_model

        self.logger.info("Best Trial for iteration %d:", iteration + 1)
        self.logger.info("Iteration %d Trial %d - Value for %s: %f", trial_iteration, self.best_trial.number + 1, config.MONITOR, best_trial_value)
        self.logger.info("Best Model Configuration:")
        for key, value in best_model_config.items():
            self.logger.info("    %s: %s", key, value)

        self.log_optuna_visualizations(study, iteration_logger, iteration)

    def _get_best_trial(self, study: Study) -> Trial:
        """
        Retrieves the best trial based on the direction of the optimization.
        """
        
        if config.DIRECTION == "minimize":
            return min(study.trials, key=lambda t: t.user_attrs.get(config.MONITOR, float("inf")))
        else:
            return max(study.trials, key=lambda t: t.user_attrs.get(config.MONITOR, float("-inf")))
    
    def _update_and_log_best_trial(self, study: Study, iteration: int):
        """
        Updates and logs the best trial information after processing optimization results.
        
        Args:
            study (Study): The Optuna study object containing the results of the optimization.
            iteration (int): The current iteration number.
        """
        
        self.best_trial = self._get_best_trial(study)

        # Check if `self.best_trial` is not None
        if self.best_trial is None:
            self.logger.info("`self.best_trial` is None.")
        else:
            # Log the contents of `self.best_trial`
            self.logger.info(f"self.best_trial: {self.best_trial}")

            # Check if the attribute `user_attrs` contains the `best_metric_value` before accessing it
            if "best_metric_value" not in self.best_trial.user_attrs:
                self.logger.error("Key 'best_metric_value' not found in best_trial user attributes.")
            else:
                # Add debug logging for the raw value
                raw_metric_value = self.best_trial.user_attrs["best_metric_value"]
                self.logger.info(f"Raw best metric value from best_trial: {raw_metric_value}")
                
                # Format the metric value
                best_trial_value = self._format_metric(raw_metric_value)

                # Log the formatted value
                self.logger.info(f"Formatted best metric value: {best_trial_value}")

        best_trial_value = self._format_metric(self.best_trial.user_attrs["best_metric_value"])
        best_model_config = self.best_trial.user_attrs["hparams"]
        best_model = self.best_trial.user_attrs["best_model"]
        trial_iteration = self.best_trial.user_attrs["iteration"]

        self.best_metric = best_trial_value
        self.best_model_config = best_model_config
        self.best_model = best_model

        # Prepare data for tabulated logging
        metrics_data = [
            [metric_name, self._format_metric(metric_value)]
            for metric_name, metric_value in self.best_trial.user_attrs.items()
            if metric_name not in ("best_model", "best_metric_value", "hparams")
        ]

        config_data = [
            [key, f"{value:.4e}" if isinstance(value, float) else value]
            for key, value in best_model_config.items()
        ]

        # Color the headers
        metrics_headers = [colored("Metric", "cyan", attrs=["bold"]), colored("Value", "cyan", attrs=["bold"])]
        config_headers = [colored("Parameter", "cyan", attrs=["bold"]), colored("Value", "cyan", attrs=["bold"])]

        self.logger.info("\n%s", "=" * 26)
        self.logger.info(colored("== Best Trial Summary ==", "magenta", attrs=["bold"]))
        self.logger.info("="*26)
        self.logger.info("Iteration: %d", iteration + 1)
        self.logger.info("Trial Number: %d", self.best_trial.number + 1)
        self.logger.info("LSTM Forward Method: %s", config.LSTM_FORWARD_METHOD)
        self.logger.info("Best Metric (%s): %s", config.MONITOR, best_trial_value)
        self.logger.info("="*26)

        # Log metrics as a table with colors and grid format
        self.logger.info("\n" + colored("Best Trial Metrics:", "magenta", attrs=["bold"]) + "\n" +
                        tabulate(metrics_data, headers=metrics_headers, tablefmt="fancy_grid"))

        # Log model configuration as a table with colors and grid format
        self.logger.info("\n" + colored("Best Model Configuration:", "magenta", attrs=["bold"]) + "\n" +
                        tabulate(config_data, headers=config_headers, tablefmt="fancy_grid"))

        self.logger.info("%s", "=" * 26)

    def _format_metric(self, metric_value):
        """
        Formats a metric value, converting tensors to scalars and formatting floats in scientific notation.
        
        Args:
            metric_value: The metric value to format.

        Returns:
            str: The formatted metric value.
        """
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()
        
        if isinstance(metric_value, float):
            return f"{metric_value:.4e}"
        
        return str(metric_value)

    def log_optuna_visualizations(self, study: Study, iteration_logger: TensorBoardLogger, iteration: int):
        """
        Log visualization graphics for Optuna study results to TensorBoard.

        This function generates visualizations for parameter importances, optimization history,
        and other relevant metrics, and logs them to TensorBoard for detailed analysis.

        Args:
            study (Study): The Optuna study object containing the trials and results.
            iteration_logger (TensorBoardLogger): The logger for the current iteration.
            iteration (int): The iteration number of the current hyperparameter optimization process.
        """

        step = len(study.trials)

        # Create directories for the current run and iteration
        vis_dir = Path(PathConfig.LOG_DIR) / "visualizations" / f"iteration_{iteration + 1}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the list of hyperparameters used in all trials
        all_params = set()
        param_pairs_count = defaultdict(int)  # Dictionary to count occurrences of parameter pairs
        
        for trial in study.trials:
            all_params.update(trial.params.keys())
            # Count occurrences of each combination of hyperparameters
            for pair in combinations(trial.params.keys(), 2):
                sorted_pair = tuple(sorted(pair))  # Sort the pair to ensure consistency
                param_pairs_count[sorted_pair] += 1  # Increment the count for this pair

        if len(study.trials) <= 1:
            self.logger.warning("Not enough trials to generate visualizations. At least 2 trials are required.")
            return
        
        # Generate and log visualizations
        visualizations = {
            "Parameter Importance": lambda: ov.plot_param_importances(study, params=list(all_params)),
            "Parallel Coordinate Plot": lambda: ov.plot_parallel_coordinate(study, params=list(all_params)),
            "Optimization History": lambda: ov.plot_optimization_history(study),
            "Slice Plot": lambda: ov.plot_slice(study, params=list(all_params)),
            "Correlation Plot": lambda: ov.plot_param_importances(study, params=list(all_params))
        }

        for vis_name, vis_func in visualizations.items():
            try:
                fig = vis_func()
                img_bytes = pio.to_image(fig, format="png")
                img_array = np.array(Image.open(io.BytesIO(img_bytes)))
                iteration_logger.experiment.add_image(f"{vis_name}/Iteration {iteration + 1} Total Trials {step}", img_array, step, dataformats="HWC")
                fig.write_image(vis_dir / f"{vis_name.replace(' ', '_').lower()}_total_trials_{step}.png")
            except Exception as e:
                self.logger.error("Failed to generate %s visualization: %s", vis_name, str(e))

        # Generate and log contour plots for each pair of hyperparameters
        for (param_i, param_j), count in param_pairs_count.items():
            if count > 1:
                try:
                    fig_contour = ov.plot_contour(study, params=[param_i, param_j])
                    contour_img = pio.to_image(fig_contour, format="png")
                    img_array = np.array(Image.open(io.BytesIO(contour_img)))
                    iteration_logger.experiment.add_image(
                        f"Contour Plot/Iteration {iteration + 1} Total Trials {step}/{param_i} vs {param_j} (Count: {count})",
                        img_array,
                        step,
                        dataformats="HWC"
                    )
                    fig_contour.write_image(vis_dir / f"contour_plot_{param_i}_{param_j}_count_{count}_total_trials_{step}.png")
                except Exception as e:
                    self.logger.error("Failed to generate contour plot for %s vs %s: %s", param_i, param_j, str(e))