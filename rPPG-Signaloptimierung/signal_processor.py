"""
This module provides functionality for processing signals, specifically for filtering and denoising pulse signals using various techniques.

Classes:
    SignalProcessor: Provides methods to process signals, including detrending, filtering, and denoising.
"""

import os
from typing import List, Tuple, Optional
import numpy as np
from datetime import datetime
from PyEMD import EMD
from scipy.signal import butter, cheby2, filtfilt, hilbert, find_peaks
from scipy.fft import fft, fftfreq
import pywt
import matplotlib.pyplot as plt
from rppg_facepatches.video_processor import path_config


class SignalProcessor:
    """
    A class to process signals, specifically for filtering and denoising pulse signals using various techniques.

    Methods:
        detrend(signal):
            Remove the mean value from the signal to detrend it.
        hilbert_normalize(signal):
            Normalize the signal using the Hilbert transform.
        bandstop_filter(signal):
            Apply a bandstop filter to the signal.
        bandpass_filter(signal):
            Apply a bandpass filter to the signal.
        wavelet_denoise(signal):
            Denoise the signal using wavelet decomposition.
        bandstop_wavelet_filter(signal):
            Apply a bandstop filter using wavelet transformation.
        bandpass_wavelet_filter(signal):
            Apply a bandpass filter using wavelet transformation.
        spatial_averaging(frame, mask):
            Perform spatial averaging on the frame using the provided mask.
        temporal_normalization(color_values, start, end):
            Normalize the temporal segment of color signals.
        compute_snr(original_signal, filtered_signal, signal_type):
            Compute the Signal-to-Noise Ratio (SNR) of a signal.
        extract_hr(pulse_signal):
            Extract the heart rate from the pulse signal using FFT.
        normalize_color_value(color_values, epsilon):
            Normalize the oRGB values to have zero mean and unit variance.
        convert_rgb_to_orbg(rgb_pixel):
            Convert an RGB pixel to oRGB color space.
    """

    def __init__(
        self,
        fps: float,
        lowcut: float = 0.8, # 48 BPM
        highcut: float = 2.5, # 150 BPM
        filter_method: str = "fft_bandstop_wt", # fft_bandstop_wt
        filter_order: int = 2, # 2
        wavelet: Optional[str] = 'rbio3.1', # rbio3.1
        wavelet_level: int = 5, # 5
        snr_method: str = "decibels",
        rs: int = 50,
    ):
        """
        Initializes the SignalProcessor class.

        Args:
            fps (float): Frames per second of the video.
            lowcut (float): Low cutoff frequency for filtering.
            highcut (float): High cutoff frequency for filtering.
            filter_order (int): Order of the filter.
            wavelet (str): Wavelet type for wavelet transformation.
            wavelet_level (int): Level of wavelet decomposition.
            snr_method (str): Method for computing SNR. Options: 'ratio', 'decibels', 'power', 'voltage', 'cov'.
        """
        self.fps = fps
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.snr_method = snr_method
        self.filter = filter_method
        self.rs = rs

    def _get_methods(self) -> List[str]:
        """
        Get the list of filtering methods.

        Returns:
            List[str]: List of filtering methods.
        """
        return [
            "bandstop",
            "bandpass",
            'cheby2_bandpass',
            'cheby2_bandstop',
            "fft_bandpass",
            "fft_bandstop",

            "bandstop_wt",
            "bandpass_wt",
            "cheby2_bandstop_wt",
            "cheby2_bandpass_wt",
            "fft_bandpass_wt",
            "fft_bandstop_wt",

            "bandstop_ht",
            "bandpass_ht",
            "cheby2_bandstop_ht",
            "cheby2_bandpass_ht",
            "fft_bandpass_ht",
            "fft_bandstop_ht",

            "bandstop_hht",
            "bandpass_hht",
            "cheby2_bandstop_hht",
            "cheby2_bandpass_hht",
            "fft_bandpass_hht",
            "fft_bandstop_hht",
        ]
    
    def _get_filtered_signal_old(self, signal: np.ndarray, filter_method: str) -> Tuple[np.ndarray, Optional[float]]:
        """
        Get the filtered signal using the specified method.

        Args:
            signal (np.ndarray): The raw pulse signal.
            filter_method (str): The filter method to be used.

        Returns:
            Tuple[np.ndarray, Optional[float]]: The filtered signal and the heart rate (if applicable).
        """
        # Methods that return both signal and heart rate
        transformation_methods = {
            "bandstop_wt": self.wavelet_transform(self.bandstop_filter(signal)),
            "bandpass_wavelet": self.wavelet_transform(self.bandpass_filter(signal)),
            "cheby2_bandstop_wavelet": self.wavelet_transform(self.cheby2_bandstop_filter(signal)),
            "cheby2_bandpass_wavelet": self.wavelet_transform(self.cheby2_bandpass_filter(signal)),
            "bandstop_hilbert": self.hilbert_transform(self.bandstop_filter(signal)),
            "bandpass_hilbert": self.hilbert_transform(self.bandpass_filter(signal)),
            "cheby2_bandstop_hilbert": self.hilbert_transform(self.cheby2_bandstop_filter(signal)),
            "cheby2_bandpass_hilbert": self.hilbert_transform(self.cheby2_bandpass_filter(signal)),
            "bandstop_hilbert_huang": self.hilbert_huang_transform(self.bandstop_filter(signal)),
            "bandpass_hilbert_huang": self.hilbert_huang_transform(self.bandpass_filter(signal)),
            "cheby2_bandstop_hilbert_huang": self.hilbert_huang_transform(self.cheby2_bandstop_filter(signal)),
            "cheby2_bandpass_hilbert_huang": self.hilbert_huang_transform(self.cheby2_bandpass_filter(signal)),
            "bandpass_fft": self.bandpass_fft(signal),
            "bandstop_fft": self.bandstop_fft(signal),
        }

        # Methods that return only the filtered signal
        filter_methods = {
            "bandstop": self.bandstop_filter(signal),
            "bandpass": self.bandpass_filter(signal),
            "cheby2_bandstop": self.cheby2_bandstop_filter(signal),
            "cheby2_bandpass": self.cheby2_bandpass_filter(signal),
        }

        if filter_method in transformation_methods:
            return transformation_methods[filter_method]
        
        if filter_method in filter_methods:
            heart_rate = self.calculate_heart_rate(filter_methods[filter_method])
            return filter_methods[filter_method], heart_rate
        
        raise ValueError(f"Unknown filter method: {filter_method}")

    def _get_filtered_signal(self, signal: np.ndarray, filter_method: str) -> Tuple[np.ndarray, Optional[float]]:
        """
        Get the filtered signal using the specified method.

        Args:
            signal (np.ndarray): The raw pulse signal.
            filter_method (str): The filter method to be used.

        Returns:
            Tuple[np.ndarray, Optional[float]]: The filtered signal and the heart rate (if applicable).
        """

        # Methods that return both signal and heart rate
        transformation_methods = {
            "bandstop_wt",
            "bandpass_wt",
            "cheby2_bandstop_wt",
            "cheby2_bandpass_wt",

            "bandstop_ht",
            "bandpass_ht",
            "cheby2_bandstop_ht",
            "cheby2_bandpass_ht",

            "bandstop_hht",
            "bandpass_hht",
            "cheby2_bandstop_hht",
            "cheby2_bandpass_hht",

            "fft_bandpass",
            "fft_bandstop",
        }

        fft_filter_methods = {
            "fft_bandpass_wt",
            "fft_bandstop_wt",
            "fft_bandpass_ht",
            "fft_bandstop_ht",
            "fft_bandpass_hht",
            "fft_bandstop_hht",
        }

        # Methods that return only the filtered signal
        filter_methods = {
            "bandstop",
            "bandpass",
            "cheby2_bandstop",
            "cheby2_bandpass",
        }

        if filter_method in transformation_methods:
            if filter_method == "bandstop_wt":
                return self.wavelet_transform(self.bandstop_filter(signal))
            elif filter_method == "bandpass_wt":
                return self.wavelet_transform(self.bandpass_filter(signal))
            elif filter_method == "cheby2_bandstop_wt":
                return self.wavelet_transform(self.cheby2_bandstop_filter(signal))
            elif filter_method == "cheby2_bandpass_wt":
                return self.wavelet_transform(self.cheby2_bandpass_filter(signal))
            elif filter_method == "bandstop_ht":
                return self.hilbert_transform(self.bandstop_filter(signal))
            elif filter_method == "bandpass_ht":
                return self.hilbert_transform(self.bandpass_filter(signal))
            elif filter_method == "cheby2_bandstop_ht":
                return self.hilbert_transform(self.cheby2_bandstop_filter(signal))
            elif filter_method == "cheby2_bandpass_ht":
                return self.hilbert_transform(self.cheby2_bandpass_filter(signal))
            elif filter_method == "bandstop_hht":
                return self.hilbert_huang_transform(self.bandstop_filter(signal))
            elif filter_method == "bandpass_hht":
                return self.hilbert_huang_transform(self.bandpass_filter(signal))
            elif filter_method == "cheby2_bandstop_hht":
                return self.hilbert_huang_transform(self.cheby2_bandstop_filter(signal))
            elif filter_method == "cheby2_bandpass_hht":
                return self.hilbert_huang_transform(self.cheby2_bandpass_filter(signal))
            elif filter_method == "fft_bandpass":
                return self.fft_bandpass(signal)
            elif filter_method == "fft_bandstop":
                return self.fft_bandstop(signal)

        elif filter_method in filter_methods:
            filtered_signal = None
            if filter_method == "bandstop":
                filtered_signal = self.bandstop_filter(signal)
            elif filter_method == "bandpass":
                filtered_signal = self.bandpass_filter(signal)
            elif filter_method == "cheby2_bandstop":
                filtered_signal = self.cheby2_bandstop_filter(signal)
            elif filter_method == "cheby2_bandpass":
                filtered_signal = self.cheby2_bandpass_filter(signal)
            
            heart_rate = self.calculate_heart_rate(filtered_signal)
            return filtered_signal, heart_rate
        
        elif filter_method in fft_filter_methods:
            if filter_method == "fft_bandpass_wt":
                filtered_signal, heart_rate_fft = self.fft_bandpass(signal)
                filtered_signal, heart_rate = self.wavelet_transform(filtered_signal)
            elif filter_method == "fft_bandstop_wt":
                filtered_signal, heart_rate_fft = self.fft_bandstop(signal)
                filtered_signal, heart_rate = self.wavelet_transform(filtered_signal)
            elif filter_method == "fft_bandpass_ht":
                filtered_signal, heart_rate_fft = self.fft_bandpass(signal)
                filtered_signal, heart_rate = self.hilbert_transform(filtered_signal)
            elif filter_method == "fft_bandstop_ht":
                filtered_signal, heart_rate_fft = self.fft_bandstop(signal)
                filtered_signal, heart_rate = self.hilbert_transform(filtered_signal)
            elif filter_method == "fft_bandpass_hht":
                filtered_signal, heart_rate_fft = self.fft_bandpass(signal)
                filtered_signal, heart_rate = self.hilbert_huang_transform(filtered_signal)
            elif filter_method == "fft_bandstop_hht":
                filtered_signal, heart_rate_fft = self.fft_bandstop(signal)
                filtered_signal, heart_rate = self.hilbert_huang_transform(filtered_signal)
            return filtered_signal, heart_rate
        else:
            raise ValueError(f"Unknown filter method: {filter_method}")
        
        # Add a final return in case an unexpected situation occurs
        return np.zeros_like(signal), None
    
    def detrend(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove the mean value from the signal to detrend it.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            np.ndarray: Detrended signal.
        """
        #return signal - np.mean(signal)
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Filter
    def bandstop_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a bandstop filter to the signal.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            np.ndarray: Filtered signal.
        """
        b, a = butter(
            self.filter_order,
            [self.lowcut / (0.5 * self.fps), self.highcut / (0.5 * self.fps)],
            btype="bandstop",
        )
        return filtfilt(b, a, signal)

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a bandpass filter to the signal.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            np.ndarray: Filtered signal.
        """
        b, a = butter(
            self.filter_order,
            [self.lowcut / (0.5 * self.fps), self.highcut / (0.5 * self.fps)],
            btype="bandpass",
        )
        return filtfilt(b, a, signal)

    def cheby2_bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a Chebyshev II bandpass filter to the signal.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            np.ndarray: Filtered signal.
        """
        nyquist = 0.5 * self.fps
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = cheby2(
            self.filter_order,
            self.rs,  # Stopband attenuation
            [low, high],
            btype="bandpass",
        )
        return filtfilt(b, a, signal)

    def cheby2_bandstop_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a Chebyshev II bandstop filter to the signal.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            np.ndarray: Filtered signal.
        """
        nyquist = 0.5 * self.fps
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = cheby2(
            self.filter_order,
            self.rs,  # Stopband attenuation
            [low, high],
            btype="bandstop",
        )
        return filtfilt(b, a, signal)

    def calculate_heart_rate(self, signal: np.ndarray) -> float:
        """
        Calculate the heart rate in beats per minute (BPM) from the filtered pulse signal.

        Args:
            signal (np.ndarray): The filtered pulse signal.

        Returns:
            float: The calculated heart rate in BPM.
        """
        # Step 1: Find peaks in the filtered signal
        peaks, _ = find_peaks(signal, distance=self.fps/2)  # Assumes heart rate < 2 Hz (120 BPM)

        # Step 2: Calculate the intervals between consecutive peaks
        peak_intervals = np.diff(peaks) / self.fps  # Convert to seconds

        # Step 3: Calculate the average heart rate
        if len(peak_intervals) > 0:
            avg_rr_interval = np.mean(peak_intervals)  # Average R-R interval
            heart_rate_bpm = 60.0 / avg_rr_interval    # Convert to BPM
        else:
            heart_rate_bpm = 0.0  # No peaks detected, return 0 BPM
        
        return heart_rate_bpm
    

    # Transformer
    def hilbert_normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize the signal using the Hilbert transform.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            np.ndarray: Normalized signal.
        """
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        return signal / amplitude_envelope
    
    def hilbert_transform(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply the Hilbert transform to the input signal and return the analytic signal and its amplitude envelope.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            Tuple[np.ndarray, float]: The analytic signal and the calculated heart rate in BPM.
        """
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        pulse_signal = signal / amplitude_envelope

        # amplitude_envelope = np.abs(analytic_signal)
        inst_freq = self.instantaneous_frequency(analytic_signal)
        heart_rate_bpm = np.mean(inst_freq) * 60.0  # Convert from Hz to BPM

        # Visualize the selected IMF's analytic signal
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_filename = f'ht_analytic_signal_{timestamp}.png'
        # self.visualize_analytic_signal(signal, analytic_signal, save_filename)
        # self.visualize_power_spectrum(pulse_signal, transform_method="hilbert")

        return pulse_signal, heart_rate_bpm

    def instantaneous_frequency(self, analytic_signal: np.ndarray) -> np.ndarray:
        """
        Calculate the instantaneous frequency from the analytic signal.

        Args:
            analytic_signal (np.ndarray): The analytic signal obtained from the Hilbert transform.

        Returns:
            np.ndarray: Instantaneous frequency in Hz.
        """
        # Calculate the instantaneous phase of the analytic signal
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # Calculate instantaneous frequency (Hz) and handle length difference
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.fps

        # Insert a zero at the beginning to maintain the same length as the input signal
        instantaneous_frequency = np.insert(instantaneous_frequency, 0, 0)
        
        return instantaneous_frequency
    
    def hilbert_huang_transform(self, signal: np.ndarray, filename_prefix: str, expected_hr: int = 80) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Apply the Hilbert-Huang Transform (HHT) to the input signal and return the reconstructed pulse signal and heart rate in BPM.

        Args:
            signal (np.ndarray): Input pulse signal.

        Returns:
            Tuple[np.ndarray, float]: The reconstructed pulse signal and the average heart rate in BPM.
        """
        # Step 1: Empirical Mode Decomposition (EMD)
        emd = EMD()
        imfs = emd(signal) # Intrinsic Mode Functions (IMFs)

        best_imf_index = None
        best_hr_diff = float('inf')
        best_hr_bpm: Optional[float] = None
        best_analytic_signal: Optional[np.ndarray] = None
        
        # Step 2: Process each IMF to find the one that best matches the expected HR
        for idx, imf in enumerate(imfs):
            # analytic_signal, heart_rate_bpm = self.hilbert_transform(imf)
            analytic_signal = hilbert(imf)
            inst_freq = self.instantaneous_frequency(analytic_signal)
            heart_rate_bpm = np.mean(inst_freq) * 60.0  # Convert from Hz to BPM

            hr_diff = abs(heart_rate_bpm - expected_hr)
            
            if hr_diff < best_hr_diff:
                best_hr_diff = hr_diff
                best_hr_bpm = heart_rate_bpm
                best_imf_index = idx
                best_analytic_signal = analytic_signal

            # self.visualize_analytic_signal(signal, analytic_signal, f'analytic_signal_imf{idx}.png')

        # Calculate the amplitude envelope of the best IMF's analytic signal
        amplitude_envelope = np.abs(best_analytic_signal)
        pulse_signal = signal / amplitude_envelope

        # Visualize the selected IMF's analytic signal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{filename_prefix}_hht_analytic_signal_best_imf{best_imf_index}_{timestamp}.png'
        self.visualize_analytic_signal(signal, analytic_signal, filename)
        self.visualize_power_spectrum(pulse_signal, transform_method='hilbert-huang')

        return pulse_signal, best_hr_bpm
    
    def visualize_analytic_signal(self, signal: np.ndarray, analytic_signal: np.ndarray, filename: str):
        """
        Visualize the components of the analytic signal.

        Args:
            signal (np.ndarray): The original input signal.
            analytic_signal (np.ndarray): The complex-valued analytic signal.
        """
        save_path = path_config.VIDEO_OUTPUT_DIR.parent / filename
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Calculate time in seconds based on signal length and FPS
        time = np.arange(len(signal)) / self.fps

        amplitude_envelope = np.abs(analytic_signal) # Amplitude envelope of the analytic signal
        normalized_signal = signal / amplitude_envelope
        instantaneous_phase = np.angle(analytic_signal) # Instantaneous phase in radians
        real_part = np.real(analytic_signal) # Real part of the analytic signal
        imaginary_part = np.imag(analytic_signal) # Imaginary part of the analytic signal

        plt.figure(figsize=(12, 10))

        # Plot original signal
        plt.subplot(3, 1, 1)
        plt.plot(time, signal, label='Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Original Signal')
        plt.grid(True)
        plt.legend()

        # # Plot amplitude envelope
        # plt.subplot(4, 1, 2)
        # plt.plot(time, amplitude_envelope, label='Amplitude Envelope', color='orange')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.title('Amplitude Envelope')
        # plt.grid(True)
        # plt.legend()

        # Plot normalized signal
        plt.subplot(3, 1, 2)
        plt.plot(time, normalized_signal, label='Normalized Signal', color='cyan')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Normalized Signal')
        plt.grid(True)
        plt.legend()

        # # Plot real and imaginary parts
        # plt.subplot(6, 1, 4)
        # plt.plot(time, real_part, label='Real Part', color='blue')
        # plt.plot(time, imaginary_part, label='Imaginary Part', color='red')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.title('Real and Imaginary Parts of Analytic Signal')
        # plt.grid(True)
        # plt.legend()

        # Plot instantaneous phase
        plt.subplot(3, 1, 3)
        plt.plot(time, instantaneous_phase, label='Instantaneous Phase', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase [radians]')
        plt.title('Instantaneous Phase')
        plt.grid(True)
        plt.legend()

        # # Plot in the complex plane
        # plt.subplot(6, 1, 6)
        # plt.plot(real_part, imaginary_part, label='Complex Plane', color='purple')
        # plt.xlabel('Real Part')
        # plt.ylabel('Imaginary Part')
        # plt.title('Analytic Signal in Complex Plane')
        # plt.grid(True)
        # plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, format='png')
        plt.close()


    def wavelet_transform(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Denoise the signal using wavelet decomposition.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            Tuple[np.ndarray, float]: Denoised signal and the calculated heart rate in BPM.
        """
        coeffs = pywt.wavedec(signal, self.wavelet, mode="per")
        sigma = np.median(np.abs(coeffs[-self.wavelet_level])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1:] = [pywt.threshold(i, value=uthresh, mode="soft") for i in coeffs[1:]]
        denoised_signal = pywt.waverec(coeffs, self.wavelet, mode="per")

        # Detect peaks in the denoised signal
        peaks, _ = find_peaks(denoised_signal, distance=self.fps/2)  # Assuming the heart rate is less than 2 Hz (120 BPM)
        
        # Calculate peak intervals and heart rate
        peak_intervals = np.diff(peaks) / self.fps  # Convert to seconds
        if len(peak_intervals) > 0:
            avg_rr_interval = np.mean(peak_intervals)  # Average R-R interval
            heart_rate_bpm = 60.0 / avg_rr_interval    # Convert to BPM
        else:
            heart_rate_bpm = 0.0  # No peaks detected, return 0 BPM
        
        # # Visualize the results
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.visualize_wt(signal, denoised_signal, peaks, heart_rate_bpm, f'wt_pulse_signal_{timestamp}.png')
        # self.visualize_power_spectrum(denoised_signal, transform_method='wavelet')

        return denoised_signal, heart_rate_bpm

    def visualize_wt(self, original_signal: np.ndarray, denoised_signal: np.ndarray, peaks: np.ndarray, heart_rate_bpm: float, filename: str):
        """
        Visualize the wavelet transform process including the original signal, denoised signal, and detected peaks.

        Args:
            original_signal (np.ndarray): The original input signal.
            denoised_signal (np.ndarray): The signal after wavelet denoising.
            peaks (np.ndarray): Indices of detected peaks in the denoised signal.
            heart_rate_bpm (float): The calculated heart rate in BPM.
        """
        save_path = path_config.VIDEO_OUTPUT_DIR / filename
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        time = np.arange(len(original_signal)) / self.fps

        plt.figure(figsize=(12, 10))

        # Plot original signal
        plt.subplot(4, 1, 1)
        plt.plot(time, original_signal, label='Original Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Original Signal')
        plt.grid(True)
        plt.legend()

        # Plot denoised signal
        plt.subplot(4, 1, 2)
        plt.plot(time, denoised_signal, label='Denoised Signal', color='orange')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Denoised Signal')
        plt.grid(True)
        plt.legend()

        # Plot denoised signal with detected peaks
        plt.subplot(4, 1, 3)
        plt.plot(time, denoised_signal, label='Denoised Signal', color='orange')
        plt.plot(time[peaks], denoised_signal[peaks], 'rx', label='Detected Peaks')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Denoised Signal with Detected Peaks')
        plt.grid(True)
        plt.legend()

        # Annotate heart rate
        plt.subplot(4, 1, 4)
        plt.text(0.5, 0.5, f'Estimated Heart Rate: {heart_rate_bpm:.2f} BPM', fontsize=16, ha='center', va='center')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def visualize_power_spectrum(self, signal: np.ndarray, transform_method: str):
        """
        Visualize and save the power spectrum of a given pulse signal.

        Args:
            signal (np.ndarray): The input pulse signal.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"power_spectrum_{path_config.VIDEO_INDEX:03d}_"
            f"{path_config.VIDEO_FILE_PATH.stem}_"
            f"{transform_method}_"
            f"{self.filter}_ord{self.filter_order}_"
            f"{self.wavelet}_lvl{self.wavelet_level}_"
            f"{timestamp}.png"
        )
        save_path = path_config.VIDEO_OUTPUT_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compute the FFT of the signal
        N = len(signal)
        fft_values = fft(signal)
        
        # Compute the corresponding frequencies
        freqs = fftfreq(N, 1/self.fps)
        
        # Compute the power spectrum
        power_spectrum = np.abs(fft_values)**2
        
        # Plot the power spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(freqs[:N // 2], power_spectrum[:N // 2])  # Plot only the positive frequencies
        plt.title('Power Spectrum of Pulse Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.grid(True)
        
        # Save the plot as an image file
        plt.savefig(save_path)
        plt.close()

    # Fast Fourier Transform (FFT)
    def apply_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply FFT to the input signal and return the frequency components and their amplitudes.

        Args:
            signal (np.ndarray): The input time-domain signal.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Frequencies and corresponding amplitudes.
        """
        freqs = np.fft.fftfreq(len(signal), d=1.0/self.fps)
        fft_signal = np.fft.fft(signal)

        return freqs, fft_signal

    def apply_inverse_fft(self, fft_signal: np.ndarray) -> np.ndarray:
        """
        Apply the inverse FFT to transform the signal back to the time domain.

        Args:
            fft_signal (np.ndarray): The frequency-domain signal.

        Returns:
            np.ndarray: The time-domain signal.
        """
        return np.fft.ifft(fft_signal).real

    def fft_bandpass(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply a bandpass filter using FFT to isolate the frequency range for pulse extraction and calculate heart rate in BPM.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            Tuple[np.ndarray, float]: Filtered signal and the calculated heart rate in BPM.
        """
        # Perform FFT
        freqs, fft_signal = self.apply_fft(signal)
        
        # Create a bandpass filter mask
        mask = (np.abs(freqs) >= self.lowcut) & (np.abs(freqs) <= self.highcut)
        fft_signal[~mask] = 0

        filtered_signal = self.apply_inverse_fft(fft_signal)
        
        # Calculate heart rate
        fft_magnitude = np.abs(fft_signal)
        peak_frequency = freqs[np.argmax(fft_magnitude[freqs > 0])]
        heart_rate_bpm = peak_frequency * 60.0

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.visualize_fft(signal, filtered_signal, heart_rate_bpm, freqs, fft_signal, f'fft_bandpass_pulse_signal_{timestamp}.png')
        # self.visualize_power_spectrum(filtered_signal, transform_method='fft-bandpass')

        return filtered_signal, heart_rate_bpm

    def fft_bandstop(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply a bandstop filter using FFT to remove specific frequency ranges from the signal and calculate heart rate in BPM.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            Tuple[np.ndarray, float]: Filtered signal and the calculated heart rate in BPM.
        """
        # Perform FFT
        freqs, fft_signal = self.apply_fft(signal)

        # Create a bandstop filter mask
        mask = (np.abs(freqs) < self.lowcut) | (np.abs(freqs) > self.highcut)

        fft_signal[~mask] = 0
        filtered_signal = self.apply_inverse_fft(fft_signal)

        # Calculate heart rate
        fft_magnitude = np.abs(fft_signal)
        peak_frequency = freqs[np.argmax(fft_magnitude[freqs > 0])]
        heart_rate_bpm = peak_frequency * 60.0

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.visualize_fft(signal, filtered_signal, heart_rate_bpm, freqs, fft_signal, f'fft_bandstop_pulse_signal_{timestamp}.png')
        # self.visualize_power_spectrum(filtered_signal, transform_method='fft-bandstop')

        return filtered_signal, heart_rate_bpm

    def visualize_fft(self, original_signal: np.ndarray, pulse_signal: np.ndarray, heart_rate_bpm: float, freqs: np.ndarray, fft_signal: np.ndarray, filename: str):
        """
        Visualize the original signal, pulse signal, heart rate, and amplitude over frequency before and after filtering.

        Args:
            original_signal (np.ndarray): The original input signal.
            pulse_signal (np.ndarray): The filtered pulse signal.
            heart_rate_bpm (float): The calculated heart rate.
            freqs (np.ndarray): Frequency components.
            fft_signal (np.ndarray): FFT of the original signal.
            fft_signal_filtered (np.ndarray): FFT of the filtered signal.
            save_path (str): Path to save the plot.
        """
        save_path = path_config.VIDEO_OUTPUT_DIR.parent / filename
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        time = np.arange(len(original_signal)) / self.fps

        plt.figure(figsize=(12, 18))

        # Plot original signal
        plt.subplot(4, 1, 1)
        plt.plot(time, original_signal, label='Original Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Original Signal')
        plt.grid(True)
        plt.legend()

        # Plot pulse signal
        plt.subplot(4, 1, 2)
        plt.plot(time, pulse_signal, label='Pulse Signal (Filtered)', color='orange')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Pulse Signal')
        plt.grid(True)
        plt.legend()

        # Annotate heart rate
        plt.subplot(4, 1, 3)
        plt.text(0.5, 0.5, f'Estimated Heart Rate: {heart_rate_bpm:.2f} BPM', fontsize=16, ha='center', va='center')
        plt.axis('off')

        # Plot FFT magnitude of original signal
        plt.subplot(4, 1, 4)
        plt.plot(freqs, np.abs(fft_signal), label='Original Signal FFT')
        plt.xlim(0, self.highcut * 2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('Original Signal FFT')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


    def spatial_averaging(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform spatial averaging on the frame using the provided mask.

        Args:
            frame (np.ndarray): Input video frame.
            mask (np.ndarray): Mask for the region of interest.
            is_orgb (bool): Flag to indicate if the color space is oRGB.

        Returns:
            np.ndarray: Spatially averaged color values.
        """
        masked_frame = frame * mask[:, :, None]
        mean_rgb = np.mean(masked_frame[mask > 0], axis=0)

        if path_config.METHOD == "orgb":
            return self.convert_rgb_to_orbg(mean_rgb)
        return mean_rgb

    def temporal_normalization(self, color_values: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        Normalize the temporal segment of color signals.

        Args:
            color_values (np.ndarray): Array of color signals.
            start (int): Start index of the segment.
            end (int): End index of the segment.

        Returns:
            np.ndarray: Normalized color signals.
        """
        color_values_segment = color_values[start : end + 1]
        mean_color_values_segment = np.mean(color_values_segment, axis=0)
        std_color_values_segment = np.std(color_values_segment, axis=0)
        std_color_values_segment[std_color_values_segment == 0] = 1
        return (color_values_segment - mean_color_values_segment) / std_color_values_segment

    def compute_snr(
        self, original_signal: np.ndarray, filtered_signal: np.ndarray, signal_type: str, plot_snr: bool = False,
    ) -> float:
        """
        Compute the SNR of a signal using different methods.

        Args:
            original_signal (np.ndarray): Input signal before filtering.
            filtered_signal (np.ndarray): Signal after bandstop filtering.
            method (str): Method to compute SNR. Options: 'ratio', 'decibels', 'power', 'voltage', 'cov'.

        Returns:
            float: SNR value.
        """
        noise = original_signal - filtered_signal
        signal_power = np.mean(np.abs(filtered_signal)**2)
        noise_power = np.mean(np.abs(noise)**2) # MSE
        epsilon = 1e-7
        
        if signal_power <= 0.0:
            return 0.0

        # Handle the case where noise power (MSE) is zero
        if noise_power < epsilon:
            return 1000.0  # Return a very high SNR value

        if self.snr_method == "ratio":
            snr = signal_power / noise_power
        elif self.snr_method in ["decibels", "power"]:
            snr = 10 * np.log10(signal_power / noise_power)
        elif self.snr_method == "voltage":
            snr = 20 * np.log10(np.sqrt(signal_power) / np.sqrt(noise_power))
        elif self.snr_method == "cov":
            snr = np.mean(filtered_signal) / np.std(noise)
        else:
            raise ValueError(
                "Invalid SNR method. Choose from 'ratio', 'decibels', 'power', 'voltage', 'cov'."
            )

        # Check if SNR is a complex number
        if np.iscomplex(snr):
            raise ValueError(f"SNR calculation resulted in a complex number: {snr}")

        if plot_snr:
            self.plot_snr(original_signal, filtered_signal, noise, signal_power, noise_power, snr, signal_type)

        return snr

    def plot_snr(self, original_signal: np.ndarray, filtered_signal: np.ndarray, noise: np.ndarray, signal_power: float, noise_power: float, snr: float, signal_type: str):
        """
        Plot the original signal, filtered signal, and noise, along with annotations for signal power, noise power, and SNR.

        Args:
            original_signal (np.ndarray): The original signal before filtering.
            filtered_signal (np.ndarray): The signal after filtering.
            noise (np.ndarray): The noise present in the signal.
            signal_power (float): The power of the signal.
            noise_power (float): The power of the noise.
            snr (float): The signal-to-noise ratio in decibels.
            signal_type (str): The type or identifier of the signal being processed.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(original_signal, label="Original Signal")
        plt.plot(filtered_signal, label="Filtered Signal")
        plt.plot(noise, label="Noise")
        plt.legend()
        plt.title("Signal Visualization for SNR Calculation")
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")

        # Annotate SNR and power values
        plt.annotate(
            f"SNR: {snr:.2f} dB",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
        )
        plt.annotate(
            f"Signal Power: {signal_power:.2f}",
            xy=(0.05, 0.90),
            xycoords="axes fraction",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
        )
        plt.annotate(
            f"Noise Power: {noise_power:.2f}",
            xy=(0.05, 0.85),
            xycoords="axes fraction",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
        )

        plt.tight_layout()
        plt.savefig(
            f"{signal_type}_snr_visualization_filter-order{self.filter_order}_wavelet-level{self.wavelet_level}_wavelet{self.wavelet}_snr{snr:.2f}.png"
        )
        plt.close()

    def extract_hr(self, pulse_signal: np.ndarray) -> float:
        """
        Extract the heart rate from the pulse signal using FFT.

        Args:
            pulse_signal (np.ndarray): Input pulse signal.

        Returns:
            float: Extracted heart rate in BPM.
        """
        if pulse_signal is None or not isinstance(pulse_signal, (list, np.ndarray)) or len(pulse_signal) == 0:
            return 0
        
        # Apply FFT to the preprocessed signal
        fft_values = fft(pulse_signal)
        fft_freqs = np.fft.fftfreq(len(pulse_signal), 1 / self.fps)

        # Isolate the positive frequencies
        positive_freqs = fft_freqs[: len(fft_freqs) // 2]
        positive_fft_values = np.abs(fft_values[: len(fft_values) // 2])

        # Find the peak in the FFT which corresponds to the heart rate
        peak_freq_index = np.argmax(positive_fft_values)
        peak_freq = positive_freqs[peak_freq_index]

        # Calculate the heart rate in BPM
        heart_rate = peak_freq * 60

        return heart_rate

    def normalize_color_value(self, color_values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Normalize the oRGB values to have zero mean and unit variance.

        Args:
            orgb_values (np.ndarray): The oRGB values to normalize.
            epsilon (float): A small value to prevent division by zero.

        Returns:
            np.ndarray: The normalized oRGB values.
        """
        mean = np.mean(color_values, axis=0)
        std = np.std(color_values, axis=0)

        # Calculate values minus mean
        color_values_minus_mean = color_values - mean

        # Handle the case where values - mean is zero by adding epsilon
        color_values_minus_mean = np.where(color_values_minus_mean == 0, epsilon, color_values_minus_mean)

        # Handle the case where std is very close to zero
        std_adjusted = np.where(std < epsilon, epsilon, std)

        normalized_color_values = color_values_minus_mean / std_adjusted

        # Check for NaN values in normalized values
        # if np.any(np.isnan(normalized_color_values)):
        #     raise ValueError("NaN values found in normalized values")

        return normalized_color_values

    @staticmethod
    def convert_rgb_to_orbg(rgb_pixel: np.ndarray) -> np.ndarray:
        """
        Convert an RGB pixel to the oRGB (opponent RGB) color space with enhanced robustness to motion artifacts.

        This function performs the following steps:
        1. Applies a linear transformation to convert the RGB pixel to an initial set of chrominance-like components.
        2. Introduces motion robustness by computing differential signals from the RGB components.
        3. Combines the original chrominance components with the motion-robust signals.
        4. Computes a rotation angle based on the combined components to further separate color information.
        5. Applies a rotation matrix to derive the final oRGB components, c_rg (Red-Green) and c_yb (Yellow-Blue).

        Args:
            rgb_pixel (np.ndarray): A 1D numpy array representing an RGB pixel, with shape (3,), where each element
                                    corresponds to the Red, Green, and Blue channel values, respectively.

        Returns:
            np.ndarray: A 1D numpy array representing the oRGB components, with shape (2,). The first element is c_rg
                        (Red-Green Chrominance), and the second element is c_yb (Yellow-Blue Chrominance).
        
        Example:
            >>> rgb_pixel = np.array([128, 64, 32])
            >>> convert_rgb_to_orbg(rgb_pixel)
            array([c_rg_value, c_yb_value])
        """
        # Step 1: Apply initial linear transformation
        linear_transform = np.array(
            [
                [0.2990, 0.5870, 0.1140],
                [0.5000, 0.5000, -1.000],
                [0.8660, -0.8660, 0.000],
            ]
        )
        _, c1, c2 = np.dot(linear_transform, rgb_pixel)

        # Step 2: Compute the rotation angle
        theta = np.arctan2(c2, c1)
        if theta < np.pi / 3:
            thetao = (3 / 2) * theta
        elif np.pi >= theta >= np.pi / 3:
            thetao = np.pi / 2 + (3 / 4) * (theta - np.pi / 3)
        else:
            thetao = theta

        # Step 3: Apply rotation matrix to separate color components
        rotation_matrix = np.array(
            [
                [np.cos(thetao - theta), -np.sin(thetao - theta)],
                [np.sin(thetao - theta), np.cos(thetao - theta)],
            ]
        )
        c_rg, c_yb = np.dot(rotation_matrix, [c1, c2])
        return np.array([c_rg, c_yb])