"""
This module provides functionality for processing pulse signals from video frames using various signal processing techniques.

Classes:
    PulseSignalProcessor: Inherits from SignalProcessor and provides methods to process pulse signals.
"""

import os
import warnings
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from multiprocessing import Pool, cpu_count
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
import torch
import torch.nn as nn

from rppg_facepatches.video_processor import path_config
from rppg_facepatches.video_processor.signal_processor import SignalProcessor
from rppg_facepatches.video_processor.snr_optimizer import SNROptimizer

warnings.filterwarnings("ignore", category=RuntimeWarning)

class PulseSignalProcessor(SignalProcessor):
    """
    A class to process pulse signals from video frames using various signal processing techniques.

    Methods:
        calculate_pulse_signals_for_segment(segment_data):
            Calculate pulse signals for segments based on the process type.

        compute_rgb_signals(signal_matrix):
            Compute pulse signals for RGB signals.

        compute_orgb_signals(signal_matrix):
            Compute pulse signals for oRGB signals.

        process_pulse_signals(pulse_signals):
            Process pulse signals including detrending, filtering, and SNR optimization.

        calculate_pulse_signals(video_frames, masks):
            Calculate pulse signals from video frames and masks.
    """

    def __init__(
        self,
        fps: float,
        pulse_signals_keys: List[str],
        window_length: Optional[int] = None,
    ):
        """
        Initializes the PulseSignalProcessor class.

        Args:
            fps (float): Frames per second of the video.
            pulse_signals_keys (List[str]): List of keys for different pulse signal variants.
            window_length (Optional[int], optional): Length of the processing window. Defaults to 1.6 * fps if not provided.
        """
        super().__init__(fps)
        self.pulse_signals_keys = pulse_signals_keys
        self.window_length = (
            window_length if window_length is not None else (int(1.6 * fps))
        )  # Default window_length based on a 1.6 second window if not provided, Window Length (in Frames) = Window Length (in Sekunden) × FPS

    def calculate_pulse_signals_for_segment_old(self, segment_data: Dict[str, List[np.ndarray]]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, dict[str, float]], Dict[str, dict[str, float]]]:
        """
        Calculate pulse signals for segments based on the process type.

        Args:
            segment_data (Dict[str, List[np.ndarray]]): The video frames and masks for each region.
            process_type (str): The type of process ('patch', 'mask' or 'background').
            is_orgb (bool): Whether to use oRGB method.

        Returns:
            Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, np.ndarray]]: A dictionary of pulse signals, SNR values, and averaged color values.
        """
        segment_pulse_signals: Dict[str, Dict[str, np.ndarray]] = {
            region: {key: np.array([]) for key in self.pulse_signals_keys}
            for region in segment_data.keys()
            if region != "frames"
        }
        color_values_dict: Dict[str, np.ndarray] = {}
        snr_values_dict: Dict[str, dict[str, float]] = {}
        hr_values_dict: Dict[str, dict[str, float]] = {}

        for region, region_masks in segment_data.items():
            if region == "frames":
                continue
            if path_config.ROI_TYPE in ["background", "mask"]:
                pulse_signals, color_values_dict[region], snr_values_dict[region], hr_values_dict[region] = self.calculate_pulse_signals(segment_data["frames"], region_masks)
                for signal_type in self.pulse_signals_keys:
                    segment_pulse_signals[region][signal_type] = pulse_signals[signal_type]

            elif path_config.ROI_TYPE == "patch":
                reordered_patches = [[patch[patch_idx] for patch in region_masks]for patch_idx in range(len(region_masks[0]))]
                patch_pulse_signals, color_values_dict[region], snr_values_dict[region], hr_values_dict[region] = zip(*[self.calculate_pulse_signals(segment_data["frames"], patch, patch_idx) for patch_idx, patch in enumerate(reordered_patches)])
                for signal_type in self.pulse_signals_keys:
                    segment_pulse_signals[region][signal_type] = np.array([patch[signal_type] for patch in patch_pulse_signals])

        # Convert lists to arrays not needed?
        # segment_pulse_signals = {key: {region: np.array(signals) for region, signals in regions.items()} for key, regions in segment_pulse_signals.items()}

        return segment_pulse_signals, color_values_dict, snr_values_dict, hr_values_dict
    
    @staticmethod
    def process_region(region_masks, frames, is_patch, pulse_signals_keys, pulse_signal_processor):
        """
        Processes a specific region of video frames to calculate pulse signals and related metrics.

        This method calculates pulse signals for a given region, either using individual patches or the entire mask,
        based on the specified region of interest type. It computes the pulse signals, color values, signal-to-noise
        ratio (SNR), and heart rate (HR) values for each region.

        Args:
            region_masks (List[np.ndarray]): The masks for the regions within the frames, each as a separate mask.
            frames (List[np.ndarray]): The video frames to process.
            is_patch (bool): Whether to process each region as a collection of patches.
            pulse_signals_keys (List[str]): Keys for the different pulse signal types to compute.
            pulse_signal_processor: An instance of the PulseSignalProcessor used for signal calculations.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, float], Dict[str, float]]:
                - pulse_signals: A dictionary of computed pulse signals for each signal type.
                - color_values: Computed color values for the processed region.
                - snr_values: Signal-to-noise ratio values for the computed signals.
                - hr_values: Heart rate values derived from the computed signals.
        """
        if is_patch:
            reordered_patches = [[patch[patch_idx] for patch in region_masks] for patch_idx in range(len(region_masks[0]))]
            patch_results = [pulse_signal_processor.calculate_pulse_signals(frames, patch, patch_idx) for patch_idx, patch in enumerate(reordered_patches)]
            patch_pulse_signals, color_values, snr_values, hr_values = zip(*patch_results)
            pulse_signals = {signal_type: np.array([patch[signal_type] for patch in patch_pulse_signals]) for signal_type in pulse_signals_keys}
        else:
            pulse_signals, color_values, snr_values, hr_values = pulse_signal_processor.calculate_pulse_signals(frames, region_masks)
        return pulse_signals, color_values, snr_values, hr_values

    def calculate_pulse_signals_for_segment_mp(self, segment_data: Dict[str, List[np.ndarray]]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, dict[str, float]], Dict[str, dict[str, float]]]:
        """
        Calculate pulse signals for each region in a video segment using multiprocessing.

        This method processes video segments to compute pulse signals across different regions. It uses multiprocessing
        to parallelize the computation of pulse signals, color values, SNR, and HR values, enhancing performance when
        handling large datasets.

        Args:
            segment_data (Dict[str, List[np.ndarray]]): A dictionary containing frames and masks for each region. The
                keys are region names and the values are lists of masks corresponding to each video frame.

        Returns:
            Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
                - segment_pulse_signals: A dictionary mapping each region to its computed pulse signals for various signal types.
                - color_values_dict: A dictionary of averaged color values for each region.
                - snr_values_dict: A dictionary mapping each region to its SNR values for each signal type.
                - hr_values_dict: A dictionary mapping each region to its HR values for each signal type.
        """
        segment_pulse_signals: Dict[str, Dict[str, np.ndarray]] = {
            region: {key: np.array([]) for key in self.pulse_signals_keys}
            for region in segment_data.keys()
            if region != "frames"
        }
        color_values_dict: Dict[str, np.ndarray] = {}
        snr_values_dict: Dict[str, dict[str, float]] = {}
        hr_values_dict: Dict[str, dict[str, float]] = {}

        with Pool(cpu_count()) as pool:
            results = []
            regions = [region for region in segment_data.keys() if region != "frames"]
            for region in regions:
                region_masks = segment_data[region]
                is_patch = path_config.ROI_TYPE == "patch"
                results.append(pool.apply_async(PulseSignalProcessor.process_region, (region_masks, segment_data["frames"], is_patch, self.pulse_signals_keys, self)))

            for region, async_result in zip(regions, results):
                pulse_signals, color_values, snr_values, hr_values = async_result.get()
                color_values_dict[region] = color_values
                snr_values_dict[region] = snr_values
                hr_values_dict[region] = hr_values
                for signal_type in self.pulse_signals_keys:
                    segment_pulse_signals[region][signal_type] = pulse_signals[signal_type]


        return segment_pulse_signals, color_values_dict, snr_values_dict, hr_values_dict
    
    def calculate_pulse_signals_for_segment(self, segment_data: Dict[str, List[np.ndarray]], use_attention: bool = False) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, dict[str, float]], Dict[str, dict[str, float]]]:
        """
        Calculate pulse signals for segments based on the process type.

        Args:
            segment_data (Dict[str, List[np.ndarray]]): The video frames and masks for each region.
            process_type (str): The type of process ('patch', 'mask' or 'background').
            is_orgb (bool): Whether to use oRGB method.

        Returns:
            Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, np.ndarray]]: A dictionary of pulse signals, SNR values, and averaged color values.
        """
        segment_pulse_signals: Dict[str, Dict[str, np.ndarray]] = {
            region: {key: np.array([]) for key in self.pulse_signals_keys}
            for region in segment_data.keys()
            if region != "frames"
        }
        color_values_dict: Dict[str, np.ndarray] = {}
        snr_values_dict: Dict[str, dict[str, float]] = {}
        hr_values_dict: Dict[str, dict[str, float]] = {}

        for region, region_masks in segment_data.items():
            if region == "frames":
                continue
            if path_config.ROI_TYPE in ["background", "mask"]:
                pulse_signals, color_values_dict[region], snr_values_dict[region], hr_values_dict[region] = self.calculate_pulse_signals(segment_data["frames"], region_masks, region)
                for signal_type in self.pulse_signals_keys:
                    segment_pulse_signals[region][signal_type] = pulse_signals[signal_type]

            elif path_config.ROI_TYPE == "patch":
                num_patches = len(region_masks[0])
                if use_attention:
                    attention_model = SpaceTimeAttentionModel(input_dim=1, embedding_dim=128, num_heads=16, num_patches=num_patches)
                reordered_patches = [[patch[patch_idx] for patch in region_masks]for patch_idx in range(num_patches)]
                patch_pulse_signals, color_values_dict[region], snr_values_dict[region], hr_values_dict[region] = zip(*[self.calculate_pulse_signals(segment_data["frames"], patch_mask, region, patch_idx) for patch_idx, patch_mask in enumerate(reordered_patches)])
                # signal_type_dict = {}
                for signal_type in self.pulse_signals_keys:
                    # Combine pulse signals for this signal_type across patches
                    combined_pulse_signals = np.array([patch[signal_type] for patch in patch_pulse_signals])

                    if use_attention:
                        # Reshape the combined signals to match the expected input shape for the model
                        # Input shape should be (seq_len, num_patches, input_dim)
                        # In this case, input_dim = 1 because each signal is a 1-dimensional value
                        # model_input = combined_pulse_signals.transpose(1, 0)  # Shape becomes (seq_len, num_patches)
                        model_input = combined_pulse_signals # shape (8, 128) (num_patches, seq_len)
                        model_input = model_input.transpose(1, 0) # shape (128, 8) (seq_len, num_patches)
                        model_input = model_input[..., np.newaxis]  # shape (128, 8, 1) (seq_len, num_patches, input_dim)
                        
                        # Apply the attention model
                        attention_output = attention_model(torch.from_numpy(model_input).float()) # torch.Size([128, 1])
                        
                        # Save the result as the final pulse signal for this region and signal_type
                        segment_pulse_signals[region][signal_type] = attention_output.detach().numpy()
                        signal_type_dict[signal_type] = self.extract_hr(segment_pulse_signals[region][signal_type])
                    else:
                        # Use the combined signals without attention
                        segment_pulse_signals[region][signal_type] = combined_pulse_signals

        return segment_pulse_signals, color_values_dict, snr_values_dict, hr_values_dict

    def compute_rgb_signals(self, signal_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute pulse signals for RGB signals.

        Args:
            S (np.ndarray): The RGB signals matrix.

        Returns:
            Dict[str, np.ndarray]: A dictionary of computed pulse signal variants.
        """
        # Define a small epsilon to avoid division by zero
        epsilon = 1e-8

        # Calculate standard deviations
        std_r = np.std(signal_matrix[0])
        std_g = np.std(signal_matrix[1])
        std_b = np.std(signal_matrix[2])

        # Add epsilon to std if it is zero or NaN
        std_r = std_r if std_r != 0 else epsilon
        std_g = std_g if std_g != 0 else epsilon
        std_b = std_b if std_b != 0 else epsilon

        # Standard RGB combinations
        signal_variants = {
            "R": signal_matrix[0],
            "G": signal_matrix[1],
            "B": signal_matrix[2],
            "RG": signal_matrix[0] + (std_r / std_g) * signal_matrix[1],
            "RB": signal_matrix[0] + (std_r / std_b) * signal_matrix[2],
            "GB": signal_matrix[1] + (std_g / std_b) * signal_matrix[2],
            "RGB": signal_matrix[0] + (std_r / std_g) * signal_matrix[1] + (std_r / std_b) * signal_matrix[2],
        }

        return {key: signal - np.mean(signal) for key, signal in signal_variants.items()}

    def compute_orgb_signals(self, signal_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute pulse signals for oRGB signals.

        Args:
            S (np.ndarray): The oRGB signals matrix.

        Returns:
            Dict[str, np.ndarray]: A dictionary of computed pulse signal variants.
        """
        # Define a small epsilon to avoid division by zero
        epsilon = 1e-8

        # Calculate standard deviations
        std_org = np.std(signal_matrix[0])
        std_oyb = np.std(signal_matrix[1])

        # Add epsilon to std if it is zero or NaN
        std_org = std_org if std_org != 0 else epsilon
        std_oyb = std_oyb if std_oyb != 0 else epsilon
        
        signal_variants = {
            "oRG": signal_matrix[0],
            "oYB": signal_matrix[1],
            "oRGYB": signal_matrix[0] + (std_org / std_oyb) * signal_matrix[1],
        }

        return {key: signal - np.mean(signal) for key, signal in signal_variants.items()}

    def process_pulse_signals(self, pulse_signals: Dict[str, np.ndarray], region: str, patch_idx: Optional[int] = None, optimize: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
        """
        Process pulse signals including detrending, filtering, and SNR optimization.

        Args:
            pulse_signals (Dict[str, np.ndarray]): The raw pulse signals.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]: Processed pulse signals, SNR values, and heart rate values.
        """
        snr_values = {key: 0.0 for key in pulse_signals.keys()}
        hr_values = {key: 0.0 for key in pulse_signals.keys()}
        hht_hr_values = {key: 0.0 for key in pulse_signals.keys()}

        if optimize:
            # Create an instance of SNROptimizer and optimize SNR
            optimizer = SNROptimizer(self.fps)
            optimizer.optimize_snr(pulse_signals, region, patch_idx, use_mp=True)

        for key in pulse_signals.keys():
            # Store the original raw signal for visualization
            raw_signal = pulse_signals[key]

            # Step 1: Detrend the signal
            signal = self.detrend(pulse_signals[key])
            pulse_signals[key] = signal

            # Step 2: Apply filtering and transformation
            pulse_signals[key], hr_values[key] = self._get_filtered_signal(pulse_signals[key], self.filter)

            # Step 3: Calculate SNR after processing
            snr_values[key] = self.compute_snr(signal, pulse_signals[key], key)

            # Step 4: Visualize raw and processed signals for comparison
            filename = f"{path_config.ROI_TYPE}_{region}_{key}_{self.filter}_{self.filter_order}_{self.wavelet}_{self.wavelet_level}_signal_comparison"
            title = f"Signal Comparison for {key} - Filter: {self.filter} (Order: {self.filter_order}), Wavelet: {self.wavelet} (Level: {self.wavelet_level})"
            
            # Plot the original raw and processed signals
            self._plot_signals(raw_signal, pulse_signals[key], title=title, filename=filename)

            # hht_pulse_signal, hht_hr_values[key] = self.hilbert_huang_transform(pulse_signals[key], key)
            # self._plot_signals(raw_signal, hht_pulse_signal, title='Hilbert-Huang-Transform of Processed Signal', filename=f'{path_config.ROI_TYPE}_{region}_{key}_{self.filter}_{self.filter_order}_{self.wavelet}_{self.wavelet_level}_hht_signal')

        return pulse_signals, snr_values, hr_values

    def _plot_signals(self, raw_signal: np.ndarray, processed_signal: np.ndarray, title: str, filename: str):
        """
        Plot raw and processed pulse signals.

        Args:
            raw_signal (np.ndarray): The raw input pulse signal.
            processed_signal (np.ndarray): The processed pulse signal.
            title (str): The title of the plot.
            filename (str): The filename to save the plot.
        """
        # Generate a timestamp string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Append the timestamp to the filename
        filename = f"{filename}_{timestamp}.png"

        # Construct the full save path
        save_path = path_config.VIDEO_OUTPUT_DIR.parent / 'rPPG_signals' / filename
        
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Calculate time in seconds using self.fps
        time = np.arange(len(raw_signal)) / self.fps

        # Detect peaks in the processed signal (considering them as heartbeats)
        peaks, _ = find_peaks(processed_signal, distance=self.fps/2.0)  # Distance is set to avoid detecting peaks too close to each other

        # Calculate Heart Rate (BPM) if there are peaks detected
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / self.fps  # Time between consecutive peaks
            average_rr_interval = np.mean(peak_intervals)  # Average time between R-R intervals
            heart_rate_bpm = 60.0 / average_rr_interval  # Convert R-R interval to BPM
            title += f" | Estimated HR: {heart_rate_bpm:.2f} BPM"
        else:
            heart_rate_bpm = None

        # Plot raw and processed signals
        plt.figure(figsize=(12, 6))
        plt.plot(time, raw_signal, label='Raw Signal', color='blue')
        plt.plot(time, processed_signal, label='Processed Signal', color='red')

        # Highlight the heartbeats on the processed signal
        # plt.plot(time[peaks], processed_signal[peaks], 'ro', label=' DetectedHeartbeats')

        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to the specified file path
        plt.savefig(save_path, format='png')
        plt.close()

    def calculate_pulse_signals(self, video_frames: List[np.ndarray], masks: List[np.ndarray], region: str, patch_idx: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, float], Dict[str, float]]:
        """
        Calculate pulse signals from video frames and masks.

        Args:
            video_frames (List[np.ndarray]): The video frames.
            masks (List[np.ndarray]): The masks for the frames.
            patch_idx (Optional[int]): The index of the patch if ROI type is 'patch'.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, float], Dict[str, float]]: The pulse signals, color values, SNR values, and heart rate values.
        """
        num_frames = len(video_frames)
        color_values = np.zeros((num_frames, 2 if path_config.METHOD == "orgb" else 3))
        pulse_signals = {signal_type: np.zeros(num_frames) for signal_type in self.pulse_signals_keys}

        epsilon = 1e-6  # Small value to avoid division by zero
        sr_failed = False  # Flag to track if 2SR calculation has failed
        apos_failed = False  # Flag to track if aPOS calculation has failed

        for n in range(num_frames):
            color_values[n] = self.spatial_averaging(video_frames[n], masks[n])

            if n >= self.window_length - 1:
                m = n - self.window_length + 1
                normalized_colors = self.temporal_normalization(color_values, m, n)
                signal_matrix = normalized_colors.T
                
                if not sr_failed:
                    try:
                        # 2SR Method Implementation
                        Ck = np.dot(signal_matrix, signal_matrix.T) / signal_matrix.shape[1]
                        Λk, Uk = np.linalg.eigh(Ck)

                        if m > 0:
                            C_prev = np.dot(color_values[m-1:n].T, color_values[m-1:n]) / (n - m + 1)
                            Λ_prev, U_prev = np.linalg.eigh(C_prev)

                            R = np.dot(Uk.T, U_prev)

                            sin_θ12 = np.sin(np.arccos(R[0, 1]))
                            if path_config.METHOD == 'orgb':
                                S = np.sqrt(Λk[0]) / np.sqrt(Λ_prev[1])
                                SR = S * sin_θ12
                                SR_prime = np.dot(SR, U_prev[1:2, 0].T)
                                pulse_signals['o2SR'][m:n+1] += SR_prime
                            if path_config.METHOD == 'rgb':
                                sin_θ13 = np.sin(np.arccos(R[0, 2]))
                                S = np.sqrt(Λk[0]) / np.sqrt(np.maximum(Λ_prev[1:], epsilon)) # Ensure no zero values
                                SR = S * np.array([sin_θ12, sin_θ13])
                                SR_prime = np.dot(SR, U_prev[1:3, 0].T)
                                pulse_signals['2SR'][m:n+1] += SR_prime
                    
                    except np.linalg.LinAlgError as e:
                        # print(f"Warning: Eigenvalue decomposition failed at frame {n}, skipping 2SR calculation. Error: {e}")
                        # Save an array of zeros with a length of 128 instead
                        pulse_signals['2SR'] = np.zeros(num_frames)
                        sr_failed = True
                
                if path_config.METHOD == 'rgb':
                    # POS Method Implementation
                    # Step 1: Temporal normalization
                    # Step 2: Apply POS projection matrix
                    Pp = np.array([[0, 1, -1], [-2, 1, 1]])
                    S = Pp @ signal_matrix
                    # Step 3: Alpha tuning to extract the pulse signal
                    alpha = self.pos_alpha_tuning(S[0], S[1]) # 1d-array n×m
                    pulse_signals['POS'][m:n+1] += self.pos_extract_pulse_signal(S, alpha)

                    if not apos_failed:
                        try:
                            # Check for NaN in the signals
                            if np.isnan(signal_matrix).any():
                                raise ValueError(f"NaN values detected in RGB signals at frame {n}")

                            # APOS Method Implementation
                            # Estimating the skin-tone vector using least squares
                            skin_tone_vector = self.estimate_skin_tone_vector(signal_matrix)
                            # Projecting onto the plane orthogonal to the skin-tone vector
                            projected_signals = self.project_onto_plane(signal_matrix, skin_tone_vector)
                            # Extracting the pulse signal using the alpha-tuning method
                            alpha = self.apos_alpha_tuning(projected_signals) # 2d-array n×m
                            pulse_signals['APOS'][m:n+1] += self.apos_extract_pulse_signal(projected_signals, alpha)
                        
                        except (ValueError, Exception) as e:
                            # print(f"Warning: {e}, skipping aPOS calculation.")
                            # Save an array of zeros with a length of num_frames instead
                            pulse_signals['aPOS'] = np.zeros(num_frames)
                            apos_failed = True
                    
                    # Compute pulse signals for RGB signals
                    compute_signals_fn = self.compute_rgb_signals
                
                else: # path_config.METHOD == 'orgb'
                    # Extract the pulse signal using the alpha-tuning method
                    alpha = self.pos_alpha_tuning(signal_matrix[0], signal_matrix[1])
                    pulse_signals['aT'][m:n+1] += self.pos_extract_pulse_signal(signal_matrix, alpha)
                    # Compute pulse signals for oRGB signals 
                    compute_signals_fn = self.compute_orgb_signals

                signals = compute_signals_fn(signal_matrix)
                for signal_type in signals.keys():
                    pulse_signals[signal_type][m : n + 1] += signals[signal_type]

        pulse_signals, snr_values, hr_values = self.process_pulse_signals(pulse_signals, region, patch_idx)

        # Normalize the color values
        color_values = self.normalize_color_value(color_values)

        return pulse_signals, color_values, snr_values, hr_values

    def estimate_skin_tone_vector(self, rgb_values):
        X = rgb_values[:2, :].T
        y = rgb_values[2, :]

        model = LinearRegression()
        model.fit(X, y)

        a, b = model.coef_
        c = model.intercept_

        N = np.array([1, 1, a + b + c])
        return N / np.linalg.norm(N)

    def project_onto_plane(self, rgb_values, normal_vector):
        projection_matrix = np.eye(3) - np.outer(normal_vector, normal_vector)
        return rgb_values.T.dot(projection_matrix)
    
    def apos_extract_pulse_signal(self, projected_signals, alpha):
        S1, S2 = projected_signals[:, 0], projected_signals[:, 1]
        return S1 - alpha * S2
    
    def pos_extract_pulse_signal(self, projected_signals, alpha):
        S1, S2 = projected_signals[0], projected_signals[1]
        return S1 - alpha * S2

    def apos_alpha_tuning(self, projected_signals):
        """
        Calculate alpha for the pulse signal extraction.
        
        :param projected_signals: Projected RGB signals
        :return: Alpha value
        """
        S1, S2 = projected_signals[:, 0], projected_signals[:, 1]
        return np.std(S1) / np.std(S2)

    def pos_alpha_tuning(self, S1: np.ndarray, S2: np.ndarray) -> float:
        """
        Perform alpha tuning to optimize the projection for pulse signal extraction.

        Args:
            S1 (np.ndarray): The first projected signal.
            S2 (np.ndarray): The second projected signal.

        Returns:
            float: The optimal alpha value.
        """
        # Calculate the standard deviation of the two signals
        sigma_S1 = np.std(S1)
        sigma_S2 = np.std(S2)
        
        # Compute the alpha value as the ratio of the standard deviations
        alpha = sigma_S1 / sigma_S2
        
        return alpha

class SpaceTimeAttentionModel(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, num_heads: int, num_patches: int, dropout_rate: float = 0.1):
        super(SpaceTimeAttentionModel, self).__init__()
        
        # Transformation layer to create embeddings from raw pulse signals
        self.transformation_layer = nn.Linear(input_dim, embedding_dim)
        
        # Positional Encoding
        self.position_encoding = PositionalEncoding(embedding_dim, max_seq_len=num_patches)
        
        # Multi-head attention for spatial-temporal context
        self.spatial_temporal_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        
        # Residual connection and layer normalization around the attention layer
        self.residual_connection = nn.Identity()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)

        # Additional layers to increase complexity
        self.additional_layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(embedding_dim),
        )

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, 1)

        # Apply dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, patch_pulse_signals): # torch.Size([128, 8, 1])
        # Step 1: Transform raw pulse signals
        transformed_signals = self.transformation_layer(patch_pulse_signals) # torch.Size([128, 8, 128])
        
        # Step 2: Apply positional encoding
        encoded_signals = self.position_encoding(transformed_signals) # torch.Size([128, 8, 128])
        
        # Step 3: Apply spatial-temporal attention with a residual connection
        attention_output, _ = self.spatial_temporal_attention(encoded_signals, encoded_signals, encoded_signals) # torch.Size([128, 8, 128]), torch.Size([8, 128, 128])
        # attention_output = self.dropout(attention_output)
        attention_output = self.residual_connection(attention_output + encoded_signals)
        attention_output = self.layer_norm1(attention_output)  # torch.Size([128, 8, 128])

        # Step 5: Pass through additional layers for increased complexity
        complex_output = self.additional_layers(attention_output)  # torch.Size([128, 8, 128])

        # Step 4: Apply the output layer to reduce dimensionality for each sequence step
        output_signal = self.output_layer(complex_output)  # torch.Size([128, 8, 1])
        
        # Step 5: Reshape the output to match the desired shape (128, 1)
        output_signal = output_signal.mean(dim=1)  # Mean over num_patches dimension (dim=1) -> torch.Size([128, 1])

        return output_signal

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_len: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix for positional encodings with size (max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_seq_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pe = pe.unsqueeze(0).unsqueeze(0)  # Reshape to add batch dimension and seq_len dimension

        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, num_patches, embedding_dim).

        Returns:
            torch.Tensor: Tensor with positional encodings added.
        """
        # Ensure positional encoding matches the input tensor's sequence length and batch size
        x = x + self.pe[:, :, :x.size(0), :].squeeze(0) # pe: torch.Size([1, 1, 8, 128]) x: torch.Size([128, 8, 128])
        return x # self.dropout(x) # torch.Size([128, 8, 128])

if __name__ == "__main__":
    input_dim = 1  # Each raw pulse signal is a single value at each time step
    embedding_dim = 128  # project the signal to a 128-dimensional space
    num_heads = 4  # / 8: Number of attention heads
    num_patches = 6  # Number of patches in the region

    model = SpaceTimeAttentionModel(input_dim, embedding_dim, num_heads, num_patches)

    # Example input (batch_size, seq_len, num_patches, input_dim)
    patch_pulse_signals = torch.randn(128, num_patches, input_dim)  # Example tensor

    output_signal = model(patch_pulse_signals)