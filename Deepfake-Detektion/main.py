"""
Entry point for LSTM module.

This script configures, optimizes, and runs a model using specific configurations of ROI (Region of Interest) 
types, signal types, and LSTM methods. It leverages PyTorch, PyTorch Lightning, and Optuna for model training 
and hyperparameter optimization. It also ensures reproducibility by setting seeds and manages GPU resources.

Functions:
    optimize_hyperparameters: Optimizes hyperparameters using Optuna.
    configure_and_run: Configures the model for a specific ROI, signal type, and LSTM method, then runs optimization.
    main: Main function to initialize and run the training process.
"""

import warnings
import logging
import torch
import torch.multiprocessing as mp

from rppg_facepatches.LSTM.utils import test_model, set_seeds, cleanup_resources
from rppg_facepatches.LSTM.optimizer import HyperparameterOptimizer
from rppg_facepatches.LSTM import config
from rppg_facepatches.LSTM.config import PathConfig

logging.getLogger("lightning").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')

warnings.filterwarnings("ignore", category=UserWarning, message="Can't initialize NVML")


def optimize_hyperparameters(method_log_dir):
    """
    Optimizes hyperparameters for the LSTM model using Optuna over different balancing methods.

    Args:
        method_log_dir (str): Directory to store the logs from the optimization process.

    Returns:
        tuple: The best metric and configuration obtained from the hyperparameter optimization.
    """
    hyperparameter_optimizer = HyperparameterOptimizer(
        n_iterations=1, method_log_dir=method_log_dir
    )
    return hyperparameter_optimizer.optimize()


def configure_and_run(roi_type, roi_subtype, signal_type):
    """
    Configures the model with a specific ROI type, ROI subtype, and signal type, then optimizes and tests it.

    Args:
        roi_type (str): The region of interest type (e.g., 'patch', 'mask', 'background').
        roi_subtype (str): The subtype of the region of interest (e.g., 'forehead', 'full_face').
        signal_type (str): The signal type for the model (e.g., 'G', 'RGB', '2SR').

    Raises:
        ValueError: If an unknown ROI type is encountered.
    """
    config.ROI_TYPE = roi_type
    config.ROI_SUBTYPE = roi_subtype
    config.SIGNAL_TYPE = signal_type

    print(
        f"Running configuration: ROI_TYPE={roi_type}, ROI_SUBTYPE={roi_subtype}, SIGNAL_TYPE={signal_type}"
    )

    # Update paths based on the current configuration
    PathConfig.update_paths(roi_type, roi_subtype, signal_type)
    PathConfig.init_directories()

    # Define valid LSTM forward methods based on ROI_TYPE
    if config.ROI_TYPE == "patch":
        lstm_methods = [
            "mean",
            # "max",
            # "avg_pool",
            # "max_pool",
            # "adaptive_pool",
        ]
    elif config.ROI_TYPE in ["mask", "background"]:
        lstm_methods = ["last"]
    else:
        raise ValueError(f"Unknown ROI_TYPE: {config.ROI_TYPE}")

    # Iterate through all possible LSTM methods and run optimization and testing.
    for method in lstm_methods:
        config.LSTM_FORWARD_METHOD = method
        print(f"Testing LSTM forward method: {method}")

        optimize_hyperparameters(method_log_dir=PathConfig.LOG_DIR / "optimization")
        test_model()


def main():
    """
    The main entry point for the LSTM module. It initializes the directories, sets seeds for reproducibility,
    and iterates through the possible configurations of ROI types, ROI subtypes, and signal types, then
    performs hyperparameter optimization and testing for each configuration.
    """
    # Set random seeds for reproducibility.
    set_seeds(config.SEED)

    # Iterate over each ROI type and its subtypes to run model configuration and testing.
    for roi_type in config.ROI_TYPES:
        roi_subtypes = (
            config.MASK_ROI_SUBTYPES
            if roi_type in ["mask", "patch"]
            else config.BACKGROUND_ROI_SUBTYPES
        )
        for roi_subtype in roi_subtypes:
            for signal_type in ["2SR"]:  # config.RGB_SIGNAL_TYPES
                configure_and_run(roi_type, roi_subtype, signal_type)

    # Clean up resources after the run.
    cleanup_resources()


if __name__ == "__main__":
    # Set the start method for multiprocessing.
    mp.set_start_method("fork")  # 'spawn', force=True

    # Set the matrix multiplication precision to 'medium' for better performance.
    torch.set_float32_matmul_precision("medium")

    main()

# Hyperparameters: {'batch_size': 144, 'max_epochs': 2000, 'dropout': 0.23514977756306715, 'learning_rate': 0.1, 'weight_decay': 1.3382684954344736e-06, 'threshold': 0.43827042701574165, 'optimizer_name': 'Adam', 'scheduler': 'CosineAnnealingLR', 'cosine_t_max': 744, 'lstm_type': 'standard', 'lstm_layers': 224, 'use_fc_batch_norm': True, 'use_lstm_layer_norm': False, 'sampler_method': 'balanced_batch', 'pos_weights_bool': True, 'balance_method': 'undersample', 'use_dropout': False, 'optimizer': 'Adam', 'momentum': 0.0, 'nhead': 16, 'num_layers': 3, 'dim_feedforward': 1408, 'dropout_transformer': 0.1305369696556904}
# 2024-09-20 16:39:34,750 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Monitor_val_loss: 0.7086
# 2024-09-20 16:39:34,750 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Accuracy: 0.8419 ##
# 2024-09-20 16:39:34,750 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Precision: 0.8913
# 2024-09-20 16:39:34,750 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Recall: 0.9361 ##
# 2024-09-20 16:39:34,750 - rppg_facepatches.LSTM.optimizer - INFO - Validation/F1 Score: 0.9131 ##
# 2024-09-20 16:39:34,750 - rppg_facepatches.LSTM.optimizer - INFO - Validation/AUROC: 0.5189
# 2024-09-20 16:39:34,750 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Loss: 0.7086

# Hyperparameters: {'batch_size': 144, 'max_epochs': 2000, 'dropout': 0.25672053038034887, 'learning_rate': 8.317637711026709e-05, 'weight_decay': 8.954115060213147e-06, 'threshold': 0.40864314583493117, 'optimizer_name': 'Adam', 'scheduler': 'CosineAnnealingLR', 'cosine_t_max': 800, 'lstm_type': 'multi_layer', 'lstm_layers': [128, 256], 'use_fc_batch_norm': True, 'use_lstm_layer_norm': False, 'sampler_method': 'balanced_batch', 'pos_weights_bool': True, 'balance_method': 'undersample', 'use_dropout': False, 'optimizer': 'Adam', 'momentum': 0.0, 'nhead': 16, 'num_layers': 3, 'dim_feedforward': 1409, 'dropout_transformer': 0.11562422703375108}
# 2024-09-23 06:07:04,802 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Monitor_val_loss: 0.7034
# 2024-09-23 06:07:04,802 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Accuracy: 0.817
# 2024-09-23 06:07:04,803 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Precision: 0.8919 ##
# 2024-09-23 06:07:04,803 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Recall: 0.9032
# 2024-09-23 06:07:04,803 - rppg_facepatches.LSTM.optimizer - INFO - Validation/F1 Score: 0.8975
# 2024-09-23 06:07:04,803 - rppg_facepatches.LSTM.optimizer - INFO - Validation/AUROC: 0.5211 ##
# 2024-09-23 06:07:04,803 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Loss: 0.7034

# Hyperparameters: {'batch_size': 144, 'max_epochs': 2000, 'dropout': 0.21079082088231663, 'learning_rate': 1.3182567385564076e-05, 'weight_decay': 1.2252995994662009e-06, 'threshold': 0.44581116685268607, 'optimizer_name': 'Adam', 'scheduler': 'CosineAnnealingLR', 'cosine_t_max': 772, 'lstm_type': 'standard', 'lstm_layers': 224, 'use_fc_batch_norm': True, 'use_lstm_layer_norm': False, 'sampler_method': 'balanced_batch', 'pos_weights_bool': True, 'balance_method': 'undersample', 'use_dropout': False, 'optimizer': 'Adam', 'momentum': 0.0, 'nhead': 16, 'num_layers': 3, 'dim_feedforward': 1408, 'dropout_transformer': 0.12556516697435124}
# 2024-09-22 22:23:41,628 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Monitor_val_loss: 0.7027
# 2024-09-22 22:23:41,628 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Accuracy: 0.8316
# 2024-09-22 22:23:41,628 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Precision: 0.8917
# 2024-09-22 22:23:41,628 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Recall: 0.9221
# 2024-09-22 22:23:41,628 - rppg_facepatches.LSTM.optimizer - INFO - Validation/F1 Score: 0.9067
# 2024-09-22 22:23:41,628 - rppg_facepatches.LSTM.optimizer - INFO - Validation/AUROC: 0.5207
# 2024-09-22 22:23:41,629 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Loss: 0.7027

# Hyperparameters: {
# 'batch_size': 104, 'max_epochs': 2000, 'dropout': 0.21320038185821316, 'learning_rate': 3.311311214825911e-05, 'weight_decay': 1e-06, 
# 'threshold': 0.3558366850493228, 'optimizer_name': 'Adam', 'scheduler': 'CosineAnnealingLR', 'cosine_t_max': 602, 'lstm_type': 'standard', 
# 'lstm_layers': 128, 'use_fc_batch_norm': True, 'use_lstm_layer_norm': False, 'sampler_method': 'balanced_batch', 'pos_weights_bool': False, 
# 'balance_method': 'undersample', 'use_dropout': False, 'optimizer': 'Adam', 'momentum': 0.0, 'nhead': 8, 'num_layers': 3, 'dim_feedforward': 1408, 
# 'dropout_transformer': 0.13744371493829896}

# Hyperparameters: {
# 'batch_size': 152, 'max_epochs': 2000, 'dropout': 0.17070253412092432, 'learning_rate': 0.02089296130854041, 'weight_decay': 1e-06, 
# 'threshold': 0.35435560862403415, 'optimizer_name': 'Adam', 'scheduler': 'CosineAnnealingLR', 'cosine_t_max': 787, 'lstm_type': 'multi_layer', 
# 'lstm_layers': [208, 192, 176], 'use_fc_batch_norm': True, 'use_lstm_layer_norm': False, 'sampler_method': 'balanced_batch', 'pos_weights_bool': True, 
# 'balance_method': 'undersample', 'use_dropout': False, 'optimizer': 'Adam', 'momentum': 0.0, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 1408, 
# 'dropout_transformer': 0.11450977214156127}

# Hyperparameters: {
# 'batch_size': 104, 'max_epochs': 2000, 'dropout': 0.1609150494386262, 'learning_rate': 0.0004365158322401656, 'weight_decay': 1e-06, 
# 'threshold': 0.35281883600128694, 'optimizer_name': 'Adam', 'scheduler': 'CosineAnnealingLR', 'cosine_t_max': 705, 'lstm_type': 'multi_layer', 
# 'lstm_layers': [128, 144], 'use_fc_batch_norm': True, 'use_lstm_layer_norm': False, 'sampler_method': 'balanced_batch', 'pos_weights_bool': False, 
# 'balance_method': 'undersample', 'use_dropout': True, 'optimizer': 'Adam', 'momentum': 0.0, 'nhead': 8, 'num_layers': 3, 'dim_feedforward': 1408, 
# 'dropout_transformer': 0.10379579299123995}

# Hyperparameters for trial 3: {'batch_size': 88, 'max_epochs': 2000, 'learning_rate': 5.248074602497728e-06, 'threshold': 0.4798824484653568, 
# 'optimizer_name': 'Adam', 'scheduler': 'CosineAnnealingLR', 'cosine_t_max': 696, 'weight_decay': 0.001, 'lstm_type': 'multi_layer', 
# 'lstm_layers': [352, 160, 256], 'lstm_dropout': 0.24144042440267569, 'transformer_nhead': 16, 'transformer_dim_feedforward': 1408, 
# 'transformer_dropout': 0.11753769238079881, 'use_lstm_dropout': True, 'use_lstm_layer_norm': False, 'use_fc_layer_norm': False, 
# 'use_fc_batch_norm': False, 'sampler_method': 'balanced_batch', 'pos_weights_bool': False, 'balance_method': 'undersample', 'optimizer': 'AdamW', 
# 'momentum': 0.0, 'transformer_num_layers': 3}
# 2024-10-04 18:32:58,219 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Monitor_val_loss: 0.6905 (optimizer.py:746)
# 2024-10-04 18:32:58,219 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Accuracy: 0.7781 (optimizer.py:746)
# 2024-10-04 18:32:58,220 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Precision: 0.8929 (optimizer.py:746)
# 2024-10-04 18:32:58,220 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Recall: 0.8522 (optimizer.py:746)
# 2024-10-04 18:32:58,220 - rppg_facepatches.LSTM.optimizer - INFO - Validation/F1 Score: 0.8721 (optimizer.py:746)
# 2024-10-04 18:32:58,220 - rppg_facepatches.LSTM.optimizer - INFO - Validation/AUROC: 0.5239 (optimizer.py:746)
# 2024-10-04 18:32:58,220 - rppg_facepatches.LSTM.optimizer - INFO - Validation/Loss: 0.6905 (optimizer.py:746)