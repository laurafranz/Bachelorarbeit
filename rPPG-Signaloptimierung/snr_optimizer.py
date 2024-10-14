"""
This module provides the SNROptimizer class for optimizing the Signal-to-Noise Ratio (SNR) of pulse signals
using various filtering methods and wavelet transformations.

Classes:
    SNROptimizer: A class to optimize the Signal-to-Noise Ratio (SNR) of pulse signals.
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime
import json
import h5py
from multiprocessing.managers import ListProxy
from multiprocessing import Pool, Manager, cpu_count
import numpy as np
import pywt
from rppg_facepatches.video_processor import path_config
from rppg_facepatches.video_processor.signal_processor import SignalProcessor


class SNROptimizer(SignalProcessor):
    """
    A class to optimize the Signal-to-Noise Ratio (SNR) of pulse signals using various filtering methods and wavelet transformations.

    Methods:
        optimize_snr(pulse_signals, patch_idx=None, use_mp=False):
            Optimize the SNR of the provided pulse signals using various methods and wavelet transformations.
            This method can use multiprocessing to speed up the optimization.

        _optimize_snr_without_mp(pulse_signals, patch_idx=None):
            Optimize the SNR without using multiprocessing.

        _optimize_snr_with_mp(pulse_signals, patch_idx=None):
            Optimize the SNR using multiprocessing.

        _prepare_tasks(pulse_signals, snr_values, results, patch_idx):
            Prepare the list of tasks for SNR optimization.

        _get_max_wavelet_level(pulse_signals):
            Get the maximum level of wavelet decomposition.

        _process_signal_wrapper(wavelet, level, order, key, signal, methods, snr_values, results, patch_idx):
            Wrapper for processing a signal with given parameters.

        _process_signal(signal, key, methods, snr_values, results):
            Process a single pulse signal with various filtering methods.

        _process_results(results):
            Process and save the results after optimizing SNR.

        _find_best_result_within_range(results):
            Find the best result within the normal heart rate range.

        _print_best_result(best_result):
            Print the best result within the normal heart rate range.

        _save_results_to_file(results, output_path, filename):
            Save the results to a JSON file.
    """

    def __init__(self, fps: float):
        """
        Initializes the SNROptimizer class.

        Args:
            fps (float): Frames per second of the video.
        """
        super().__init__(fps)
        self.hr_low = 48
        self.hr_high = 150
        self.snr_low = 0 # 12
        self.snr_high = 60

    def optimize_snr(self, pulse_signals: Dict[str, np.ndarray], region: str, patch_idx: Optional[int] = None, use_mp: bool = False):
        """
        Optimize the SNR of the provided pulse signals using various methods and wavelet transformations.
        This method can use multiprocessing to speed up the optimization.

        Args:
            pulse_signals (Dict[str, np.ndarray]): A dictionary of raw pulse signals where keys are signal types and values are numpy arrays.
            patch_idx (Optional[int]): The index of the patch if ROI type is 'patch'. Defaults to None.
            use_mp (bool): Whether to use multiprocessing to speed up the computation. Defaults to False.
        """
        if use_mp:
            results = self._optimize_snr_with_mp(pulse_signals, region, patch_idx)
        else:
            results = self._optimize_snr_without_mp(pulse_signals, region, patch_idx)

        self._process_results(results)

    def _optimize_snr_without_mp(self, pulse_signals: Dict[str, np.ndarray], region: str, patch_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Optimize the SNR of the provided pulse signals using various methods and wavelet transformations.

        Args:
            pulse_signals (Dict[str, np.ndarray]): A dictionary of raw pulse signals.
            patch_idx (Optional[int]): The index of the patch if ROI type is 'patch'. Defaults to None.
        """
        snr_values: Dict[str, float] = {}
        results: List[Dict[str, Any]] = []
        tasks = self._prepare_tasks(pulse_signals, snr_values, results, region, patch_idx)

        for task in tasks:
            self._process_signal_wrapper(*task)

        return results

    def _optimize_snr_with_mp(self, pulse_signals: Dict[str, np.ndarray], region: str, patch_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Optimize the Signal-to-Noise Ratio (SNR) of the provided pulse signals using various methods and wavelet transformations
        in a parallelized manner.

        This method distributes the SNR optimization tasks across multiple processes to speed up the computation.
        It creates a list of tasks, each representing a combination of wavelet, wavelet level, filter order, and signal,
        and uses a multiprocessing pool to process these tasks in parallel.

        Args:
            pulse_signals (Dict[str, np.ndarray]): A dictionary of raw pulse signals where keys are signal types and values are numpy arrays.
            patch_idx (Optional[int]): The index of the patch if ROI type is 'patch'. Defaults to None.
        """
        snr_values: Dict[str, float] = {}
        manager = Manager()
        results = manager.list()
        tasks = self._prepare_tasks(pulse_signals, snr_values, results, region, patch_idx)

        with Pool(processes=cpu_count()-1) as pool:
            pool.starmap(self._process_signal_wrapper, tasks)

        return list(results)

    def _prepare_tasks(self, pulse_signals: Dict[str, np.ndarray], snr_values: Dict[str, float], results: Union[List[Dict[str, Any]], ListProxy], region: str, patch_idx: Optional[int]
                       ) -> List[Tuple[Any, int, str, int, str, np.ndarray, Dict[str, float], Union[List[Dict[str, Any]], ListProxy], str, Optional[int]]]:
        """
        Prepare the list of tasks for SNR optimization.

        Args:
            pulse_signals (Dict[str, np.ndarray]): A dictionary of raw pulse signals.
            snr_values (Dict[str, float]): Dictionary to store SNR values.
            results (List[Dict[str, Any]]): List to store the results of the optimization.
            patch_idx (Optional[int]): The index of the patch if ROI type is 'patch'.

        Returns:
            List[Tuple]: A list of tasks, each task is a tuple of parameters.
        """
        # tasks = []
        tasks: List[Tuple[Any, int, str, int, str, np.ndarray, Dict[str, float], Union[List[Dict[str, Any]], ListProxy], str, Optional[int]]] = []
        wavelets = pywt.wavelist()
        filter_orders = [2, 3, 4, 5, 6]
        filter_methods = self._get_methods()

        for order in filter_orders:
            for signal_type, signal in pulse_signals.items():
                for filter_method in filter_methods:
                    if 'wt' in filter_method:
                        for wavelet in wavelets:
                            max_level = self._get_max_wavelet_level(pulse_signals, wavelet)
                            if max_level is None:
                                continue

                            # Ensure the level is appropriate for the signal length
                            max_possible_level = pywt.dwt_max_level(data_len=len(next(iter(pulse_signals.values()))), filter_len=pywt.Wavelet(wavelet).dec_len)
                            max_level = min(max_level, max_possible_level)

                            for level in range(1, max_level + 1):
                                tasks.append((wavelet, level, filter_method, order, signal_type, signal, snr_values, results, region, patch_idx))

                    else:
                        tasks.append(('None', 0, filter_method, order, signal_type, signal, snr_values, results, region, patch_idx))

        return tasks
    
    def _process_signal_wrapper(self, 
                                wavelet: Any, 
                                wavelet_level: int,
                                filter_method: str,
                                filter_order: int,
                                key: str,
                                signal: np.ndarray,
                                snr_values: Dict[str, float],
                                results: Union[List[Dict[str, Any]], ListProxy],
                                region: str, patch_idx: int,
                                ):
        """
        A wrapper method to set the wavelet, wavelet level, and filter order before processing a signal.

        This method is used to facilitate the parallel processing of signals by setting the necessary attributes
        for each task and then calling the `_process_signal` method to perform the actual SNR optimization.

        Args:
            wavelet (str): The name of the wavelet to be used.
            wavelet_level (int): The level of the wavelet decomposition.
            filter_method (str): The filtering method to be applied.
            filter_order (int): The order of the filter to be used.
            key (str): The key representing the type of the signal.
            signal (np.ndarray): The raw pulse signal to be processed.
            snr_values (Dict[str, float]): Dictionary to store SNR values.
            results (List[Dict[str, Any]]): List to store the results of the optimization.
            region (str): The region from which the signal is extracted.
            patch_idx (Optional[int]): The index of the patch if ROI type is 'patch'.
            apply_hilbert (bool): Whether to apply Hilbert transformation.
        """
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.filter = filter_method
        self.filter_order = filter_order
        self._process_signal(signal, key, wavelet, wavelet_level, filter_method, filter_order, snr_values, results, region, patch_idx)

    def _get_max_wavelet_level(self, pulse_signals: Dict[str, np.ndarray], wavelet) -> Optional[int]:
        """
        Get the maximum level of wavelet decomposition.

        Args:
            pulse_signals (Dict[str, np.ndarray]): A dictionary of raw pulse signals.

        Returns:
            Optional[int]: The maximum wavelet level or None if an error occurs.
        """
        try:
            # Get the length of the first signal in the dictionary
            first_signal_length = next(iter(pulse_signals.values())).size
            return pywt.dwt_max_level(first_signal_length, wavelet)
        except (ValueError, StopIteration):
            return None

    def _process_signal(
        self,
        signal: np.ndarray,
        key: str,
        wavelet: Any, 
        wavelet_level: int,
        filter_method: str,
        filter_order: int,
        snr_values: Dict[str, float],
        results: Union[List[Dict[str, Any]], ListProxy],
        region: str,
        patch_idx: Optional[int],
    ):
        """
        Process a single pulse signal with various filtering methods.

        Args:
            signal (np.ndarray): The raw pulse signal.
            key (str): The key of the pulse signal.
            methods (List[str]): List of filtering methods.
            snr_values (Dict[str, float]): Dictionary to store SNR values.
            results (List[Dict[str, Any]]): List to store results.
            patch_idx (Optional[int]): The index of the patch if ROI type is 'patch'.
            apply_hilbert (bool): Whether to apply Hilbert transformation.
        """
        signal = self.detrend(signal)
        
        filtered_signal, heart_rate = self._get_filtered_signal(signal, filter_method)
        
        snr = self.compute_snr(signal, filtered_signal, key)
        snr_values[f"{key}_{filter_method}"] = snr

        results.append(
            {
                "signal_type": key,
                "heart_rate": heart_rate,
                "snr": snr,
                "wavelet": wavelet,
                "wavelet_level": wavelet_level,
                "filter": filter_method,
                "filter_order": filter_order,
                "region": region,
                "patch_idx": patch_idx if patch_idx is not None else -1,
            }
        )

    def _process_results(self, results: List[Dict[str, Any]]):
        """
        Process and save the results after optimizing SNR.

        Args:
            results (List[Dict[str, Any]]): List of results.
        """
        results = sorted(results, key=lambda x: x["snr"], reverse=True)

        best_result_within_range = self._find_best_result_within_range(results)
        if best_result_within_range:
            filename = self._generate_filename(best_result_within_range)
            self._save_results_to_file(results, filename)
            # self._print_best_result(best_result_within_range)
        else:
            self._save_results_to_file(results, 'snr_optimization')

    def _find_best_result_within_range(
        self, results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best result within the normal heart rate range.

        Args:
            results (List[Dict[str, Any]]): List of results.

        Returns:
            Optional[Dict[str, Any]]: The best result within the normal heart rate range or None.
        """
        # Filter results to those within the normal heart rate range
        filtered_results = [
            result for result in results
            if self.hr_low <= result["heart_rate"] <= self.hr_high and result["snr"] != 1000
        ]

        # If no results are within the range, return None
        if not filtered_results:
            return None

        # Find the result with the highest SNR value
        best_result = max(filtered_results, key=lambda x: x['snr'])

        return best_result

    def _generate_filename(self, best_result: Dict[str, Any]) -> str:
        """
        Generate a filename based on the best result configuration.

        Args:
            best_result (Dict[str, Any]): The best result.

        Returns:
            str: The generated filename.
        """
        filename = (
            f"snr_optimization_{best_result['region']}_patch{best_result['patch_idx']}_"
            f"signaltype{best_result['signal_type']}_hr{best_result['heart_rate']:.2f}BPM_"
            f"snr{best_result['snr']:.2f}dB_filter_{best_result['filter']}_"
            f"wavelet_{best_result['wavelet']}_wlevel_{best_result['wavelet_level']}_"
            f"forder_{best_result['filter_order']}"
        )
        return filename.replace(" ", "_").replace("/", "_")

    def _print_best_result(self, best_result: Dict[str, Any]):
        """
        Print the best result within the normal heart rate range.

        Args:
            best_result (Dict[str, Any]): The best result.
        """
        print(f"Signal Type: {best_result['signal_type']}")
        print(f"Heart Rate: {best_result['heart_rate']:.2f} BPM")
        print(f"Best SNR: {best_result['snr']:.2f} dB")
        print(f"Best Wavelet: {best_result['wavelet']}")
        print(f"Best Level: {best_result['wavelet_level']}")
        print(f"Best Method: {best_result['filter']}")
        print(f"Best Order: {best_result['filter_order']}")
        print(f"Region: {best_result['region']}")
        print(f"Patch Index: {best_result['patch_idx']}\n")

    def _save_results_to_file(self, results: List[Dict[str, Any]], filename: str):
        """
        Save the results to an HDF5 file.

        Args:
            results (List[Dict[str, Any]]): List of results.
        """
        output_path = path_config.SNR_OPTIMIZER_OUTPUT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"{filename}_{timestamp}.h5"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(results_file, "w") as f:
            for i, result in enumerate(results):
                group = f.create_group(f"result_{i}")
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value)
                    elif isinstance(value, (int, float, str)):
                        group.attrs[key] = value
                    elif isinstance(value, complex):
                        group.attrs[key] = np.complex128(value)
                    else:
                        raise TypeError(f"Unsupported data type: {type(value)}")


        # with open(results_file, "w", encoding='utf8') as f:
        #     json.dump(results, f, indent=4)