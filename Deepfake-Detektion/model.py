"""
LSTM-Based Binary Classification Module for PyTorch Lightning

This module provides classes and utilities to build, train, and evaluate LSTM-based models, specifically designed for 
binary classification tasks on sequence data. The architecture is modular, allowing flexibility in terms of LSTM configuration,
fully connected layers, and optional transformer layers for patch-based processing.

Classes:
    - LSTMLightning: Main class implementing the PyTorch Lightning module for training and evaluating the LSTM-based model.
    - MultiLayerLSTM: Implements a multi-layer LSTM with optional layer normalization and dropout.
    - PatchLSTM: A custom LSTM model for patch-based data processing with support for batch normalization in the fully connected layer.

Environment Variables:
    - TORCH_CUDA_LAUNCH_BLOCKING: Ensures CUDA operations are synchronized to debug memory allocation issues.
    - TF_ENABLE_ONEDNN_OPTS: Disables OneDNN optimizations for TensorFlow compatibility.
    - PYTORCH_CUDA_ALLOC_CONF: Sets the CUDA memory allocator configuration.
    - XLA_FLAGS: Disables logging for XLA devices to reduce log verbosity.
"""

import os
import warnings
import logging
from typing import List, Tuple, Type, Optional, Union, Any, Dict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
from torch.nn import LSTM, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryROC,
)

from rppg_facepatches.LSTM import config
from rppg_facepatches.LSTM.datamodule import DataModule

warnings.filterwarnings("ignore", category=UserWarning, message="Can't initialize NVML")
warnings.filterwarnings(
    "ignore", message="Checkpoint directory .* exists and is not empty"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="dropout option adds dropout after all but last recurrent layer",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="No positive samples in targets, true positive value should be meaningless.",
)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TORCH_CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# Disable XLA logging
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class LSTMLightning(pl.LightningModule):
    """
    A PyTorch Lightning module for training LSTM-based models, specifically designed for binary classification tasks on sequence data.

    This module allows configuring LSTM layers and additional fully connected layers, handling imbalanced datasets, and tracking various metrics
    during training, validation, and testing. It also supports patch-based processing and the application of a Transformer encoder for patch aggregation.

    Attributes:
        hparams (Dict[str, Any]): Hyperparameters for configuring the model (LSTM layers, dropout rates, learning rates, etc.).
        data_module (DataModule): DataModule instance for managing datasets and dataloaders.
        lstm (nn.Module): Configured LSTM layer(s).
        fc (nn.Module): Fully connected layer for final binary classification.
        metrics (Dict[str, MetricCollection]): A dictionary to track evaluation metrics for training, validation, and testing stages.
        test_outputs (List[torch.Tensor]): A list to store predictions during the test stage.
        test_labels (List[torch.Tensor]): A list to store true labels during the test stage.
    """

    def __init__(self, hparams: Dict[str, Any], data_module: DataModule):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Access pos_weights from DataModule
        self.data_module = data_module
        if self.data_module.pos_weights is not None:
            self.pos_weights = self.data_module.pos_weights.to(self.device)
        else:
            self.pos_weights = None

        # LSTM configurations
        self.lstm_type: str = self.hparams["lstm_type"]
        self.dropout: float = self.hparams["lstm_dropout"]
        self.threshold: float = self.hparams["threshold"]

        # Determine input dimension based on LSTM layer configuration
        if isinstance(self.hparams["lstm_layers"], int):
            self.input_dim: int = self.hparams["lstm_layers"]
        else:
            self.input_dim: int = self.hparams["lstm_layers"][-1]

        # # Adjust the input dimension based on the LSTM forward method
        # method = config.LSTM_FORWARD_METHOD
        # if method in ["mean_max", "mean_min", "min_max", "adaptive_pool"]:
        #     self.input_dim *= 2 # For concatenated methods, double the input dimension

        # Create LSTM and FC layers
        self.lstm: nn.Module = self._create_lstm(self.lstm_type)
        self.fc: nn.Module = self._create_fc() # Fully connected layer for classification
        if config.LSTM_FORWARD_METHOD == "attention":
            self.attention_layer = nn.Linear(self.input_dim, 1, bias=False) # Attention layer
        
        # Transformer encoder setup for patch-based processing
        if config.ROI_TYPE == "patch":
            self.transformer_layer = TransformerEncoderLayer(
                d_model=self.input_dim,  # Input size of the transformer
                nhead=self.hparams["transformer_nhead"],  # Number of attention heads
                dim_feedforward=self.hparams["transformer_dim_feedforward"],  # Transformer feedforward size
                dropout=self.hparams["transformer_dropout"],
                batch_first=True,  # Set batch_first to True
            )
            self.transformer_encoder = TransformerEncoder(
                self.transformer_layer, num_layers=self.hparams["transformer_num_layers"]
            )

        # Set up metrics for tracking performance
        self.metrics: Dict[str, MetricCollection] = self._setup_metrics()

        # Store test outputs and labels for logging confusion matrix and ROC curve
        self.test_outputs: List[torch.Tensor] = []
        self.test_labels: List[torch.Tensor] = []

    def _create_lstm(self, lstm_type: str) -> nn.Module:
        """
        Creates the LSTM layers based on the model configuration.

        Args:
            lstm_type (str): The type of LSTM to create ('standard', 'multi_layer').

        Returns:
            nn.Module: The configured LSTM layer(s).
        """
        num_layers = (
            len(self.hparams["lstm_layers"])
            if isinstance(self.hparams["lstm_layers"], list)
            else 1
        )
        self.dropout = self.dropout if num_layers > 1 else 0.0

        if config.ROI_TYPE == "patch":
            # Custom LSTM for patch-based processing
            return PatchLSTM(
                input_size=config.INPUT_SIZE,
                fc_input_size=self.input_dim,
                lstm_layers=self.hparams["lstm_layers"],
                dropout=self.dropout,
                lstm_type=self.lstm_type,
                use_lstm_layer_norm=self.hparams["use_lstm_layer_norm"],
                use_fc_batch_norm=self.hparams["use_fc_batch_norm"],
            )
        else:
            if lstm_type == "standard":
                # Standard single-layer LSTM
                return nn.LSTM(
                    input_size=config.INPUT_SIZE,
                    hidden_size=self.hparams["lstm_layers"],
                    dropout=self.dropout,
                    batch_first=True,
                )
            elif lstm_type == "multi_layer":
                # Multi-layer LSTM
                return MultiLayerLSTM(
                    input_size=config.INPUT_SIZE,
                    layer_type=nn.LSTM,
                    layer_sizes=self.hparams["lstm_layers"],
                    use_layer_norm=self.hparams["use_lstm_layer_norm"],
                    dropout=self.dropout,
                    batch_first=True,
                )
            return nn.LSTM(  # Fallback to single-layer LSTM
                input_size=config.INPUT_SIZE,
                hidden_size=self.hparams["lstm_layers"],
                dropout=self.dropout,
                batch_first=True,
            )

    def _create_fc(self) -> nn.Module:
        """
        Creates the fully connected layer for binary classification.

        Returns:
            nn.Module: Configured fully connected layer.
        """
        input_dim = self.input_dim

        # Apply normalization if required (LayerNorm or BatchNorm)
        if self.hparams["use_fc_batch_norm"]:
            return nn.Sequential(
                nn.BatchNorm1d(input_dim),  # BatchNorm1d across batch dimension
                nn.Linear(input_dim, 1)
            )
        elif self.hparams["use_fc_layer_norm"]:
            return nn.Sequential(
                nn.LayerNorm(input_dim),  # LayerNorm across feature dimension
                nn.Linear(input_dim, 1)
            )
        else:
            return nn.Sequential(nn.Linear(input_dim, 1))

    def _setup_metrics(self) -> Dict[str, MetricCollection]:
        """
        Sets up the metrics for tracking model performance during training, validation, and testing.

        Returns:
            Dict[str, MetricCollection]: A dictionary of metric collections for different stages.
        """
        # Metrics for binary classification
        metric_collection = MetricCollection(
            {
                "accuracy": BinaryAccuracy(threshold=self.threshold),
                "precision": BinaryPrecision(threshold=self.threshold),
                "recall": BinaryRecall(threshold=self.threshold),
                "f1_score": BinaryF1Score(threshold=self.threshold),
                "auroc": BinaryAUROC(),
            }
        )

        # Move metrics to the correct device (GPU/CPU)
        metric_collection = metric_collection.to(config.DEVICE)

        return {
            "train": metric_collection.clone(prefix="train_"),
            "val": metric_collection.clone(prefix="val_"),
            "test": metric_collection.clone(prefix="test_"),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        For patch-based processing, the model processes each patch individually through the LSTM and combines the outputs.
        For sequence processing, the model processes the entire sequence directly.

        Args:
            x (torch.Tensor): The input tensor with shape depending on whether it's patch-based or sequence-based processing.

        Returns:
            torch.Tensor: The output of the fully connected layer after processing the input through the LSTM.
        """
        method = config.LSTM_FORWARD_METHOD

        if config.ROI_TYPE == "patch":
            # Patch-based processing
            batch_size, num_patches, seq_len, input_size = x.shape # Shape: (batch_size, num_patches, seq_len, input_size)
            patches = [x[:, i, :, :] for i in range(num_patches)] # List of tensors, each of shape (batch_size, seq_len, input_size)
            patch_outputs = [self.lstm(patch) for patch in patches] # Shape: (batch_size, seq_length, input_size)
            patch_outputs = torch.stack(patch_outputs, dim=1) # Stack patch outputs, Shape: (batch_size, num_patches, seq_len, hidden_size)
            last_outputs = patch_outputs[:, :, -1, : ] # Last output from each patch, Shape: (batch_size, num_patches, hidden_size)
            transformer_out = self.transformer_encoder(last_outputs) # Shape: (batch_size, num_patches, hidden_size)
            return self._apply_method(transformer_out, method)

        # Single sequence processing
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, input_size)
        return self._apply_method(lstm_out, method)

    def _apply_method(self, lstm_out: torch.Tensor, method: str) -> torch.Tensor:
        """
        Apply the specified method to the LSTM output.

        Args:
            lstm_out (torch.Tensor): The output tensor from the LSTM.
            method (str): The method to apply to the LSTM output.

        Returns:
            torch.Tensor: The processed output tensor.
        """
        # Handle different methods to process LSTM output
        if method == "last":
            last_output = lstm_out[:, -1, :]
            return self.fc(last_output)

        elif method == "mean":
            mean_output = lstm_out.mean(dim=1)
            return self.fc(mean_output)

        elif method == "max":
            max_output = lstm_out.max(dim=1).values
            return self.fc(max_output)

        elif method == "min":
            min_output = lstm_out.min(dim=1).values
            return self.fc(min_output)

        elif method == "mean_max":
            mean_output = lstm_out.mean(dim=1)
            max_output = lstm_out.max(dim=1).values
            combined_output = torch.cat((mean_output, max_output), dim=1)
            return self.fc(combined_output)

        elif method == "mean_min":
            mean_output = lstm_out.mean(dim=1)
            min_output = lstm_out.min(dim=1).values
            combined_output = torch.cat((mean_output, min_output), dim=1)
            return self.fc(combined_output)

        elif method == "min_max":
            min_output = lstm_out.min(dim=1).values
            max_output = lstm_out.max(dim=1).values
            combined_output = torch.cat((min_output, max_output), dim=1)
            return self.fc(combined_output)

        elif method == "attention":
            attention_weights = torch.softmax(self.attention_layer(lstm_out), dim=1)
            attended_output = (attention_weights * lstm_out).sum(dim=1)
            return self.fc(attended_output)

        elif method == "avg_pool":
            avg_pool = F.adaptive_avg_pool1d(
                lstm_out.permute(0, 2, 1), output_size=1
            ).squeeze(-1)
            return self.fc(avg_pool)

        elif method == "max_pool":
            max_pool = F.adaptive_max_pool1d(
                lstm_out.permute(0, 2, 1), output_size=1
            ).squeeze(-1)
            return self.fc(max_pool)

        elif method == "adaptive_pool":
            avg_pool = F.adaptive_avg_pool1d(
                lstm_out.permute(0, 2, 1), output_size=1
            ).squeeze(-1)
            max_pool = F.adaptive_max_pool1d(
                lstm_out.permute(0, 2, 1), output_size=1
            ).squeeze(-1)
            combined_adaptive_output = torch.cat((avg_pool, max_pool), dim=1)
            return self.fc(combined_adaptive_output)

        else:
            raise ValueError(f"Unknown method '{method}' specified.")

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the binary cross-entropy loss with logits, applying a positive weight if provided.

        Args:
            logits (torch.Tensor): The raw model outputs before applying the sigmoid function.
            labels (torch.Tensor): The ground truth binary labels.

        Returns:
            torch.Tensor: The computed loss value.
        """
        if self.pos_weights is not None:
            self.pos_weights = torch.clamp(self.pos_weights, 1e-5, 1e5)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(logits, labels.float())

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, Dict[str, Any]]]:
        """
        Configures the optimizer and learning rate scheduler based on the model's hyperparameters.

        Returns:
            Dict[str, Union[Optimizer, Dict[str, Any]]]: A dictionary containing the optimizer and learning rate scheduler configuration.
        """
        optimizer_class = getattr(torch.optim, self.hparams["optimizer"])
        optimizer_params = {
            "lr": self.hparams["learning_rate"],
            "weight_decay": self.hparams["weight_decay"],
            "maximize": True if config.MODE == "max" else False,
        }
        if self.hparams["optimizer"] in ["RMSprop", "SGD"]:
            optimizer_params["momentum"] = self.hparams["momentum"]

        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        # Configure learning rate scheduler
        scheduler_config = {
            "CosineAnnealingLR": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams["cosine_t_max"]
            ),
            "ReduceLROnPlateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.MODE,
                factor=self.hparams["factor"],
                patience=self.hparams["patience"],
            ),
            "ExponentialLR": lambda: torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.hparams["gamma"]
            ),
        }
        scheduler = scheduler_config[self.hparams["scheduler"]]()
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": config.MONITOR,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _shared_step(self, batch: Tuple[Tensor, Tensor], stage: str) -> Tensor:
        """
        A shared function for training, validation, and test steps. It processes the input batch,
        computes the model's predictions, calculates the loss, and updates the metrics for the given stage.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the features (input tensor) and labels (ground truth).
            stage (str): The current stage of the model ('train', 'val', or 'test').

        Returns:
            torch.Tensor: The computed loss value for the batch.
        """
        features, labels = batch
        # # Print features and labels for debugging
        # print(f"Stage: {stage} | Features: {features.shape} | Labels: {labels.shape}")
        # print(f"Labels content (stage: {stage}): {labels}")
        # print(f"Features content (stage: {stage}): {features}")

        # Forward pass through the model
        outputs = self(features).reshape_as(labels)
        probs = torch.sigmoid(outputs)

        # Convert probabilities to binary predictions using the threshold
        binary_preds = (probs > self.threshold).float().reshape_as(labels)

        # Compute the loss for the current batch
        loss = self.loss_fn(outputs, labels.float())

        # Handle invalid loss values (e.g., NaN or Inf)
        if not torch.isfinite(loss):
            self.log(f"{stage}_loss_is_inf", True, on_step=True, on_epoch=True)
            loss = torch.tensor(1e6, device=config.DEVICE) # Use a high but finite value

        # Update metrics for the current stage
        self.metrics[stage].update(binary_preds, labels)

        # Log the loss for the current stage
        self.log(
            f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )

        # Save predictions and labels for test stage to log confusion matrix and ROC curve
        if stage == "test":
            self.test_outputs.append(binary_preds)
            self.test_labels.append(labels)

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], _: int) -> Tensor:
        """
        Executes one training step. Invoked by the PyTorch Lightning trainer for each batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data from the training DataLoader.
            _ (int): Index of the batch (not used in this function).

        Returns:
            torch.Tensor: The loss of the current training batch.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], _: int) -> Tensor:
        """
        Executes one validation step. Invoked by the PyTorch Lightning trainer for each batch during validation.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data from the validation DataLoader.
            _ (int): Index of the batch (not used in this function).

        Returns:
            torch.Tensor: The loss of the current validation batch.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], _: int) -> Tensor:
        """
        Executes one test step. Invoked by the PyTorch Lightning trainer for each batch during testing.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data from the test DataLoader.
            _ (int): Index of the batch (not used in this function).

        Returns:
            torch.Tensor: The loss of the current test batch.
        """
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        """
        Executes at the end of each training epoch.

        This method logs the accumulated training metrics and optionally clears cached data 
        to free up memory.
        """
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        """
        Executes at the end of each validation epoch.

        This method logs the accumulated validation metrics and optionally clears cached data
        to free up memory.
        """
        self._log_metrics("val")

    def on_test_epoch_end(self):
        """
        Executes at the end of the testing epoch.

        This method computes and logs the final test metrics, generates and saves the confusion matrix
        and ROC curve, and clears the stored predictions and labels.
        """
        self._log_metrics("test")
        self._log_confusion_matrix_and_roc_curve()

    def _log_metrics(self, stage: str):
        """
        Computes and logs the metrics for the specified stage (train, validation, or test).

        This method gathers all metrics for the given stage, logs them, and resets the metric states.
        If any metric value is non-finite (NaN or Inf), it is replaced with 0 to ensure stability.

        Args:
            stage (str): The current stage ('train', 'val', or 'test').
        """
        metrics: Dict[str, Tensor] = self.metrics[stage].compute()

        # Handle non-finite metric values (replace with 0)
        for name, value in metrics.items():
            if not torch.isfinite(value):
                self.log(f"{stage}_{name}_is_inf", True, on_step=False, on_epoch=True)
                metrics[name] = torch.tensor(0.0)  # Standardwert
        
        # Log all the computed metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        # Reset the metric states
        self.metrics[stage].reset()

    def _log_confusion_matrix_and_roc_curve(self):
        """
        Logs the confusion matrix and ROC curve for the test stage.

        This method generates and logs both the confusion matrix and ROC curve 
        for the predictions made during the test stage. It also clears the stored 
        test predictions and labels after logging.
        """
        preds: Tensor = torch.cat(self.test_outputs)
        labels: Tensor = torch.cat(self.test_labels)

        # Log the confusion matrix and ROC curve
        self._log_confusion_matrix(preds, labels, "test")
        self._log_roc_curve(preds, labels, "test")

        # Clear the stored predictions and labels
        self.test_outputs.clear()
        self.test_labels.clear()

    def _log_confusion_matrix(self, preds: Tensor, labels: Tensor, phase: str):
        """
        Generates and logs the confusion matrix for a given phase.

        Args:
            preds (Tensor): The predicted labels.
            labels (Tensor): The true labels.
            phase (str): The current phase ('train', 'val', or 'test').
        """
        bcm = BinaryConfusionMatrix().to(config.DEVICE) # Move the confusion matrix metric to the device
        bcm.update(preds, labels.int())
        
        # Generate confusion matrix plot
        fig, ax = bcm.plot()
        ax.set_title(f"Confusion Matrix - {phase.capitalize()}")
        self.logger.experiment.add_figure(
            f"{phase}_confusion_matrix", fig, self.current_epoch
        )

        # Log the confusion matrix as a figure
        confmat = bcm(preds, labels.int())
        self.log_matrix(confmat.cpu().numpy(), f"{phase}_confusion_matrix")

    def _log_roc_curve(self, preds: Tensor, labels: Tensor, phase: str):
        """
        Generates and logs the ROC curve for a given phase.

        Args:
            preds (Tensor): The predicted labels.
            labels (Tensor): The true labels.
            phase (str): The current phase ('train', 'val', or 'test').
        """
        roc = BinaryROC().to(config.DEVICE)  # Move the ROC to the specified device
        roc.update(preds, labels.int())

        # Generate ROC curve plot
        fig, ax = roc.plot()
        ax.set_title(f"ROC Curve - {phase.capitalize()}")
        self.logger.experiment.add_figure(f"{phase}_roc_curve", fig, self.current_epoch)

        # Calculate and log the ROC AUC score
        fpr, tpr, _ = roc_curve(labels.cpu().numpy(), preds.cpu().numpy())
        roc_auc = auc(fpr, tpr)
        self.log(
            f"{phase}_roc_auc", roc_auc, on_step=False, on_epoch=True, sync_dist=True
        )

    def log_matrix(self, confmat: np.ndarray, title: str):
        """
        Logs a confusion matrix as a figure and saves it as a PNG file.

        Args:
            confmat (np.ndarray): The confusion matrix to log.
            title (str): The title of the figure.
        """
        fig, ax = plt.subplots()
        cax = ax.matshow(confmat, interpolation="nearest", cmap=plt.get_cmap("Blues"))
        fig.colorbar(cax)

        # Annotate each cell with the corresponding value
        for (i, j), val in np.ndenumerate(confmat):
            ax.text(j, i, f"{val}", ha="center", va="center", color="black")
        
        plt.xlabel(" ")
        plt.ylabel(" ")
        plt.title(title)
        
        # Log and save the confusion matrix figure
        self.logger.experiment.add_figure(title, fig, self.current_epoch)
        plt.savefig(f"{title}.png")
        plt.close(fig)

    def cleanup(self):
        """
        Frees up resources, clears caches, and resets internal states.
        This method is intended to be called after each epoch to optimize resource usage.
        """
        self.test_outputs.clear()
        self.test_labels.clear()
        self.metrics.clear()

        # Free GPU memory if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MultiLayerLSTM(nn.Module):
    """
    A multi-layer LSTM module that supports layer normalization and dropout between layers.

    This module allows the creation of a stack of LSTM layers with optional layer normalization after each layer.
    Dropout can be applied between layers to prevent overfitting, and initial hidden states for each layer can be created dynamically.

    Args:
        input_size (int): The number of expected features in the input sequence.
        layer_type (Type[LSTM]): The type of LSTM layer to use (e.g., nn.LSTM).
        layer_sizes (List[int], optional): A list of integers specifying the hidden sizes for each LSTM layer. Defaults to [64, 64].
        dropout (float, optional): The dropout rate to apply between LSTM layers. Defaults to 0.5.
        use_layer_norm (bool, optional): Whether to apply layer normalization after each LSTM layer. Defaults to False.

    Attributes:
        layers (nn.ModuleList): A list of LSTM layers, each created according to the specified `layer_sizes`.
        layer_norms (nn.ModuleList): A list of LayerNorm modules for applying layer normalization after each LSTM layer.
        input_size (int): The number of expected features in the input sequence.
        layer_sizes (List[int]): The hidden sizes for each LSTM layer.
        use_layer_norm (bool): Whether to apply layer normalization after each LSTM layer.
        dropout (float): The dropout rate applied between LSTM layers.
    """

    def __init__(
        self,
        input_size: int,
        layer_type: Type[LSTM],
        layer_sizes: List[int] = [64, 64],
        dropout: float = 0.5,
        use_layer_norm: bool = False,
        *args,
        **kwargs,
    ):
        super(MultiLayerLSTM, self).__init__()

        self.use_layer_morm = use_layer_norm # Whether to apply layer normalization after each LSTM layer
        self.dropout = dropout # Dropout rate between LSTM layers

        self.layers = nn.ModuleList() # List to store multiple LSTM layers
        self.layer_sizes = layer_sizes # Store the layer sizes
        self.input_size = input_size # Store the input size

        # Create LSTM layers based on the given sizes
        for size in layer_sizes:
            layer = layer_type(
                input_size=input_size, hidden_size=size, dropout=self.dropout, **kwargs
            )
            self.layers.append(layer)
            input_size = size # Update the size for the next layer

        # Create layer normalization modules if required
        self.layer_norms = nn.ModuleList([nn.LayerNorm(size) for size in layer_sizes])

    def reset_parameters(self):
        """
        Resets the parameters of all LSTM layers.

        This function is useful to reinitialize the weights of the LSTM layers,
        typically called before training starts or when reinitializing the model.
        """
        for layer in self.layers:
            layer.reset_parameters()

    def create_hiddens(self, batch_size: int = 1) -> List[Tuple[Tensor, Tensor]]:
        """
        Creates the initial hidden and cell states for each LSTM layer.

        This function initializes the hidden and cell states for all LSTM layers based on the given batch size.

        Args:
            batch_size (int, optional): The batch size. Defaults to 1.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples, each containing the hidden and cell states for a layer.
        """
        hiddens = []
        for layer in self.layers:
            num_directions = 2 if layer.bidirectional else 1 # Account for bidirectional LSTM
            # Initialize hidden and cell states as zeros
            hiddens.append(
                (
                    torch.zeros(
                        layer.num_layers * num_directions,
                        batch_size,
                        layer.hidden_size,
                        device=config.DEVICE,
                    ),
                    torch.zeros(
                        layer.num_layers * num_directions,
                        batch_size,
                        layer.hidden_size,
                        device=config.DEVICE,
                    ),
                )
            )
        return hiddens

    def sample_mask(self):
        """
        Samples masks for each LSTM layer if applicable.

        If the LSTM layer has a `sample_mask` method, this method is invoked to sample masks.
        This is useful when implementing techniques like dropout.
        """
        for layer in self.layers:
            if hasattr(layer, "sample_mask"):
                layer.sample_mask()

    def forward(
        self,
        x: torch.Tensor,
        hiddens: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Performs a forward pass through the multi-layer LSTM.

        This method processes the input tensor `x` through each LSTM layer in sequence.
        Optionally applies layer normalization after each layer if specified.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_size).
            hiddens (Optional[List[Tuple[torch.Tensor, torch.Tensor]]], optional): The initial hidden and cell states for each LSTM layer.
                If not provided, they will be initialized. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]: 
                - The output tensor after passing through all LSTM layers.
                - The updated hidden and cell states for each LSTM layer.
        """
        # If hidden states are not provided, initialize them
        if hiddens is None:
            hiddens = self.create_hiddens(x.size(0))

        new_hiddens = [] # Store the updated hidden and cell states

        # Pass the input through each LSTM layer sequentially
        for layer, layer_norm, h in zip(self.layers, self.layer_norms, hiddens):
            x, new_h = layer(x, h) # Forward pass through the LSTM layer
            if self.use_layer_morm:
                x = layer_norm(x) # Apply layer normalization if enabled
            new_hiddens.append(new_h) # Store the updated hidden and cell states
        
        return x, new_hiddens # Return the final output and updated hidden states


class PatchLSTM(nn.Module):
    """
    A custom LSTM-based module designed to process sequence data, with support for multiple LSTM layers and
    an optional fully connected layer with batch normalization.

    This module allows for flexible LSTM configurations, including the ability to specify the number of LSTM layers,
    apply dropout between layers, and optionally use layer normalization or batch normalization.

    Args:
        input_size (int): The number of expected features in the input sequence.
        fc_input_size (int): The input size for the fully connected layer.
        lstm_layers (Union[int, List[int]]): The number and size(s) of hidden layers in the LSTM.
            Can be a single integer for a single LSTM layer, or a list of integers for multiple layers.
        dropout (float): Dropout probability to apply between LSTM layers. If only one layer is specified, dropout is ignored.
        lstm_type (str): The type of LSTM to use, either 'standard' or 'multi_layer'.
        use_lstm_layer_norm (bool, optional): Whether to apply layer normalization after each LSTM layer. Defaults to True.
        use_fc_batch_norm (bool, optional): Whether to apply batch normalization before the fully connected layer. Defaults to True.

    Attributes:
        lstm_type (str): The type of LSTM used ('standard' or 'multi_layer').
        input_size (int): The number of features expected in the input sequence.
        lstm_layers (Union[int, List[int]]): The number and size(s) of hidden layers in the LSTM.
        use_lstm_layer_norm (bool): Whether layer normalization is applied after each LSTM layer.
        dropout (float): The dropout probability applied between LSTM layers.
        fc_input_size (int): The input size for the fully connected layer.
        use_fc_batch_norm (bool): Whether batch normalization is applied before the fully connected layer.
        lstm (nn.Module): The LSTM module created based on the configuration.
        fc (nn.Module): The fully connected layer created based on the configuration.
    """

    def __init__(
        self,
        input_size: int,
        fc_input_size: int,
        lstm_layers: Union[int, List[int]],
        dropout: float,
        lstm_type: str,
        use_lstm_layer_norm: bool = True,
        use_fc_batch_norm: bool = True,
    ):
        super(PatchLSTM, self).__init__()

        # Initialize LSTM and FC configuration based on the given arguments
        self.lstm_type = lstm_type
        self.input_size = input_size
        self.lstm_layers = lstm_layers
        self.use_lstm_layer_norm = use_lstm_layer_norm
        self.dropout = dropout
        self.fc_input_size = fc_input_size
        self.use_fc_batch_norm = use_fc_batch_norm

        # Create the LSTM module based on the configuration
        self.lstm = self._create_lstm()

        # Create the fully connected layer with optional batch normalization
        self.fc = self._create_fc()

    def _create_lstm(self) -> nn.Module:
        """
        Creates the LSTM module based on the model configuration.

        If the LSTM type is 'multi_layer', a custom `MultiLayerLSTM` module is created.
        Otherwise, a standard PyTorch LSTM is used.

        Returns:
            nn.Module: The created LSTM module.
        """
        if self.lstm_type == "multi_layer":
            # Multi-layer LSTM with custom layer sizes and optional layer normalization
            return MultiLayerLSTM(
                input_size=self.input_size,
                layer_type=nn.LSTM,
                layer_sizes=self.lstm_layers,
                use_layer_norm=self.use_lstm_layer_norm,
                dropout=self.dropout,
                batch_first=True,
            )
        else:
            # Standard LSTM with a single or multiple hidden layers
            return nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.lstm_layers,
                num_layers=(
                    1 if isinstance(self.lstm_layers, int) else len(self.lstm_layers)
                ),
                dropout=(
                    0
                    if isinstance(self.lstm_layers, int) and self.lstm_layers == 1
                    else self.dropout
                ),
                batch_first=True,
            )

    def _create_fc(self) -> nn.Module:
        """
        Creates the fully connected layer, with optional batch normalization applied before the linear layer.

        Returns:
            nn.Module: The fully connected layer with optional batch normalization.
        """
        if self.use_fc_batch_norm:
            # Apply batch normalization before the fully connected layer
            return nn.Sequential(
                nn.BatchNorm1d(self.fc_input_size),
                nn.Linear(self.fc_input_size, self.fc_input_size),
            )
        else:
            # Standard fully connected layer without batch normalization
            return nn.Sequential(
                nn.Linear(self.fc_input_size, self.fc_input_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM and fully connected layer (if applicable).

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: The output of the LSTM (or optionally the fully connected layer output).
        """
        # Pass the input through the LSTM layer(s)
        lstm_out, _ = self.lstm(x)
        return lstm_out
