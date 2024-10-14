"""
This module provides classes and methods for analyzing Signal-to-Noise Ratio (SNR) and Heart Rate (HR) data from video frames.
The main class, `SNRAnalyzer`, includes functions to load, process, analyze, and visualize SNR and HR metrics across different 
categories and configurations, facilitating the evaluation of signal quality in various scenarios.

Key Features:
    - Load and aggregate data from multiple directories and regions of interest (ROI).
    - Calculate and analyze SNR and HR scores based on different configurations.
    - Perform statistical analysis to identify significant differences in metrics across categories.
    - Generate detailed visualizations, including dual-axis boxplots and significance bar plots.
    - Identify the best combinations of signal processing parameters based on a weighted score.
    - Save analysis results and visualizations for further examination and reporting.

Usage Example:
    snr_analyzer = SNRAnalyzer()
    
    subfolders = ['real/train', 'real/val', 'real/test']
    roi_types = ['mask', 'patch']

    snr_analyzer.process_and_save_results(subfolders, roi_types)
    snr_analyzer.load_and_aggregate_results(subfolders, roi_types)

    df = snr_analyzer.load_data(subfolders, roi_types)
    snr_analyzer.analyze_best_combinations(df, top_n=3)
"""

import os
import gc
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from glob import glob
import h5py
import pandas as pd
import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy import stats
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from scipy.stats import shapiro, kruskal, norm
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from math import sqrt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from rppg_facepatches.video_processor import path_config

matplotlib.use('agg') # Non-interactive backend for file output

# Ensure that all plots display correctly
sns.set(style="whitegrid")


class SNRAnalyzer:
    """
    A class to analyze Signal-to-Noise Ratio (SNR) data from video frames.

    Methods:
        load_results_from_directories(base_dir, filename_pattern):
            Load results from JSON files in the specified directory and its subdirectories.

        save_results_to_file(results, filename):
            Save the results to a JSON file.

        load_data(subfolders, roi_types):
            Load and combine data from multiple subfolders and ROI types into a single DataFrame.

        calculate_combined_stats(group):
            Calculate the combined mean, combined standard deviation, and mean range for a grouped DataFrame.

        calculate_snr_scores(df, category):
            Calculate Signal-to-Noise Ratio (SNR) scores and related statistics for a specified category.

        calculate_hr_scores(df, category):
            Calculate heart rate (HR) scores and related statistics for a specified category.

        calculate_combined_scores(df, category, snr_weight, hr_weight, frequency_weight):
            Calculate and combine SNR, HR, and frequency scores into a final weighted score for a specified category.

        analyze_category(df, category, output_dir, top_n, snr_weight, hr_weight):
            Analyze and identify the best categories based on combined SNR, HR, and composite scores.

        plot_dual_axis_boxplot(df_filtered, y_column1, y_column2, filename_suffix, category, output_dir, highlight_category):
            Create a dual-axis plot with a boxplot and scatter plot for visualizing two metrics across different categories.

        get_top_n_combinations(df, category, snr_weight, hr_weight, frequency_weight, top_n):
            Identify and retrieve the top N combinations of signal processing parameters based on a weighted score.

        check_statistical_significance(df, category, top_combinations, alpha):
            Check the statistical significance of signal processing metrics (SNR and Heart Rate) across different categories.

        visualize_statistical_significance(significance_results, output_dir):
            Visualize the statistical significance of different metrics (SNR and Heart Rate) across various categories.

        analyze_best_combinations(df, top_n):
            Analyze and identify the best combinations of signal processing parameters across various categories.
    """
    
    def __init__(self, hr_low: int = 60, hr_high: int = 100, snr_low: int = 0, snr_high: int = 60):
        self.hr_low = hr_low
        self.hr_high = hr_high
        self.snr_low = snr_low # 12
        self.snr_high = snr_high
    
    def process_and_save_results(self):
        """
        Process and save results for each combination of subfolder and ROI type.

        This function iterates over the provided subfolders and ROI types, loads the results from JSON files,
        and saves the aggregated results into separate JSON files, organized by subfolder and ROI type.
        """
        
        base_output_dir = path_config.SNR_ANALYZER_OUTPUT_DIR.parent.parent.parent.parent
        for subfolder in ['real/train', 'real/val', 'real/test']:
            for roi_type in ['mask', 'patch', 'background']:
                # Update path configurations for the current subfolder and ROI type
                path_config.SUBFOLDER = subfolder
                path_config.ROI_TYPE = roi_type
                path_config.update_paths()

                # Get files organized by parent folder
                files_by_parent_folders = get_files_by_parent_folder(path_config.SNR_OPTIMIZER_OUTPUT_DIR.parent, 'snr_optimization')
                
                # Iterate over each parent folder
                for parent_folder, result_files in files_by_parent_folders.items():
                    print(f"Processing files in parent folder: {parent_folder}")

                    # Construct the filename pattern to check if any file starts with this pattern
                    filename_prefix = f"results_{subfolder.replace('/', '_')}_{roi_type}_{parent_folder.stem}"
                    existing_files = glob(os.path.join(base_output_dir, f"{filename_prefix}*.csv"))
                    
                    # Check if any file with the prefix already exists
                    if existing_files:
                        print(f"Files with prefix {filename_prefix} already exist. Skipping processing for {subfolder}, {roi_type}, {parent_folder.stem}.")
                        continue

                    # Load results from the specified directory using the given filename pattern
                    results: List[Dict[str, Any]] = self.load_results_from_directories(
                        result_files,
                        parent_folder.stem,
                        path_config.SNR_OPTIMIZER_OUTPUT_DIR.parent,
                    )
                    
                    # Construct a filename to save the results
                    results_output_filename = f"{filename_prefix}.csv"
                    results_output_path = os.path.join(base_output_dir, results_output_filename)
                    self.save_results_to_csv_file(results, results_output_path)
                    
                    # Clear memory after processing each subfolder/ROI type combination
                    del results
                    gc.collect()

    def load_results_old(self, filename_pattern: str) -> pd.DataFrame:
        """
        Load and aggregate results from CSV files matching a given pattern.

        This function searches the base directory for CSV files matching the specified pattern,
        loads their contents into a pandas DataFrame, and aggregates the results.

        Args:
            filename_pattern (str): The pattern to match filenames (e.g., 'results_*').

        Returns:
            pd.DataFrame: A DataFrame containing the aggregated results.
        """
        
        base_dir = path_config.SNR_ANALYZER_OUTPUT_DIR.parent.parent.parent.parent
        filename_pattern = '*' + filename_pattern + '*.csv'
        
        # Get all result files matching the pattern
        result_files = list(base_dir.rglob(filename_pattern))

        if not result_files:
            print(f"No files found matching the pattern {filename_pattern}.")
            return pd.DataFrame()
    
        all_results = []

        for batch_file in result_files:
            try:
                # Read the CSV file in chunks to handle large files efficiently
                for chunk in pd.read_csv(batch_file, chunksize=10000):
                    all_results.append(chunk)
                    gc.collect()  # Collect garbage after each chunk
            except Exception as e:
                print(f"Error reading {batch_file}: {e}")
                continue

        # Concatenate all chunks into a single DataFrame
        df_results = pd.concat(all_results, ignore_index=True)
        
        # Clear memory after processing
        del all_results
        gc.collect()

        return df_results

    def load_results(self, filename_pattern: str) -> pd.DataFrame:
        """
        Load and aggregate results from CSV files matching a given pattern.

        Args:
            filename_pattern (str): The pattern to match filenames (e.g., 'results_*').

        Returns:
            pd.DataFrame: A DataFrame containing the aggregated results.
        """
        base_dir = path_config.SNR_ANALYZER_OUTPUT_DIR.parent.parent.parent.parent / 'patch_results'
        filename_pattern = '*' + filename_pattern + '*.csv'

        result_files = list(base_dir.rglob(filename_pattern))

        if not result_files:
            print(f"No files found matching the pattern {filename_pattern}.")
            return pd.DataFrame()

        all_results = []

        with tqdm(total=len(result_files), desc="Processing files", position=0, leave=True) as pbar:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(process_file, file): file for file in result_files}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        print(f"Error processing file {futures[future]}: {e}")
                    finally:
                        # Garbage collection to free memory
                        del future
                        gc.collect()

                    pbar.update(1)

        if all_results:
            df_results = pd.concat(all_results, ignore_index=True)
        else:
            df_results = pd.DataFrame()

        # Final garbage collection to free memory
        del all_results
        gc.collect()

        return df_results

    def load_results_from_directories_old(self, base_dir: Union[str, Path], filename_pattern: str) -> List[Dict[str, Any]]:
        """
        Load results from HDF5 files in the specified directory and its subdirectories.

        Args:
            base_dir (Union[str, Path]): The base directory to search for result files.
            filename_pattern (str): The pattern to match filenames (e.g., 'snr_optimization_*.h5').

        Returns:
            List[Dict[str, Any]]: A list of results loaded from HDF5 files.
        """
        base_dir = Path(base_dir)
        filename_pattern = filename_pattern + '*.h5'
        result_files = list(base_dir.rglob(filename_pattern))
        all_results = []

        def _recursively_load_group(group: h5py.Group, result_container: Union[List[Dict[str, Any]], Dict[str, Any]]):
            """
            Recursively load the contents of an HDF5 group into a list of dictionaries or a dictionary.

            Args:
                group (h5py.Group): The HDF5 group to load.
                result_container (Union[List[Dict[str, Any]], Dict[str, Any]]): The container (list or dict) to store the loaded data.
            """
            sub_group_dict: Dict[str, Any] = {}
            sub_group_list = []

            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    sub_group_dict[key] = item[()]  # Load the dataset into the dictionary
                elif isinstance(item, h5py.Group):
                    sub_group: Dict[str, Any] = {}
                    _recursively_load_group(item, sub_group)
                    sub_group_list.append(sub_group)
                    
            # Load attributes as well
            if group.attrs:
                for key, value in group.attrs.items():
                    sub_group_dict[key] = value

                # Get the video name based on the file path
                file_path = Path(group.file.filename)
                sub_group_dict['video_name'] = f"{file_path.parent.name}"
                sub_group_dict['roi_type'] = f"{file_path.parent.parent.name}"
                sub_group_dict['subfolder'] = f"{file_path.parent.parent.parent.parent.parent.name}_{file_path.parent.parent.parent.parent.name}"

            # Only append to the result list once the entire sub_group_dict is populated
            if isinstance(result_container, list):
                result_container = sub_group_list
                return result_container
            elif isinstance(result_container, dict):
                result_container.update(sub_group_dict)

        for result_file in result_files:
            with h5py.File(result_file, 'r') as f:
                result_list: List[Dict[str, Any]] = []
                result_list = _recursively_load_group(f, result_list)  # Start at the root of the file
                all_results.extend(result_list)

        return all_results

    def load_results_from_directories_old2(self, base_dir: Union[str, Path], filename_pattern: str) -> List[Dict[str, Any]]:
        """
        Load results from HDF5 files in the specified directory and its subdirectories.

        Args:
            base_dir (Union[str, Path]): The base directory to search for result files.
            filename_pattern (str): The pattern to match filenames (e.g., 'snr_optimization_*.h5').

        Returns:
            List[Dict[str, Any]]: A list of results loaded from HDF5 files.
        """
        base_dir = Path(base_dir)
        filename_pattern = filename_pattern + '*.h5'
        result_files = list(base_dir.rglob(filename_pattern))
        all_results = []

        def _load_group(group: h5py.Group) -> Dict[str, Any]:
            """
            Load the contents of an HDF5 group into a dictionary.

            Args:
                group (h5py.Group): The HDF5 group to load.

            Returns:
                Dict[str, Any]: A dictionary containing the group's data.
            """
            group_dict = {key: item[()] for key, item in group.items() if isinstance(item, h5py.Dataset)}
            group_dict.update({key: group.attrs[key] for key in group.attrs})
            return group_dict

        def _process_file(result_file: Path) -> List[Dict[str, Any]]:
            """
            Process a single HDF5 file and return its contents as a list of dictionaries.

            Args:
                result_file (Path): The path to the HDF5 file.

            Returns:
                List[Dict[str, Any]]: A list of dictionaries containing the file's data.
            """
            results = []
            with h5py.File(result_file, 'r') as f:
                for key in f.keys():
                    group = f[key]
                    result_dict = _load_group(group)

                    # Add metadata
                    file_path = Path(f.filename)
                    result_dict['video_name'] = file_path.parent.name
                    result_dict['roi_type'] = file_path.parent.parent.name
                    result_dict['subfolder'] = f"{file_path.parent.parent.parent.parent.parent.name}_{file_path.parent.parent.parent.parent.name}"

                    results.append(result_dict)
            return results

        # Use threading to speed up the file loading process
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(_process_file, result_file): result_file for result_file in result_files}

            for future in as_completed(future_to_file):
                all_results.extend(future.result())

        return all_results

    def load_results_from_directories(self, result_files, parent_folder, base_dir: Union[str, Path], batch_size: int = 100) -> List[Dict[str, Any]]:
        base_dir = Path(base_dir)
        temp_dir = Path(base_dir) / "temp_results"
        temp_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        # Process files in batches
        for i in tqdm(range(0, len(result_files), batch_size), desc="Processing batches", position=0):
            batch_files = result_files[i:i + batch_size]
            batch_filename = temp_dir / f"{parent_folder}_batch_{i // batch_size + 1}.csv"

            # Check if the batch file already exists
            if batch_filename.exists():
                print(f"Batch file {batch_filename} already exists. Skipping processing for this batch.")
                continue

            batch_results = []

            with tqdm(total=len(batch_files), desc=f"Processing batch {i//batch_size + 1}", position=1, leave=False) as pbar:

                with ProcessPoolExecutor() as executor:
                    futures = {executor.submit(_process_file, file): file for file in batch_files}
                    
                    # Use tqdm to display progress
                    for future in as_completed(futures):
                        result = future.result()  # Get the result from the future
                        batch_results.extend(result)
                        del result  # Explicitly delete to free memory
                        del future  # Optionally delete the future object to free memory
                        gc.collect()
                        
                        pbar.update(1)

            # Save the batch results to a CSV file
            df_batch = pd.DataFrame(batch_results)
            df_batch.to_csv(batch_filename, index=False)

            # Clear batch_results to free memory
            del batch_results, df_batch
            gc.collect()

        # Load all batch CSV files for the current parent folder and combine results
        for batch_file in temp_dir.glob(f"{parent_folder}_batch_*.csv"):
            df_batch = pd.read_csv(batch_file)
            all_results.extend(df_batch.to_dict(orient='records'))
            del df_batch
            # Optionally delete the batch file after loading
            # batch_file.unlink()

        # Clean up the temporary directory
        # temp_dir.rmdir()

        return all_results

    def save_results_to_file(self, results: List[Dict[str, Any]], filename: Union[str, Path]):
        """
        Save the results to an HDF5 file.

        Args:
            results (List[Dict[str, Any]]): List of results to be saved.
            filename (Union[str, Path]): The base filename or full path for the output file.
        """
        
        # Convert filename to Path object if it is a string
        if isinstance(filename, str):
            filename = Path(path_config.SNR_ANALYZER_OUTPUT_DIR) / filename
        
        # Ensure the directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract the stem and suffix, then append the timestamp
        new_filename = filename.with_name(f"{filename.stem}_{timestamp}.h5")

        try:
            # Write the results to the HDF5 file
            with h5py.File(new_filename, "w") as f:
                for i, result in enumerate(results):
                    group = f.create_group(f"result_{i}")
                    for key, value in result.items():
                        if isinstance(value, np.ndarray):
                            group.create_dataset(key, data=value)
                        elif isinstance(value, list) or isinstance(value, dict):
                            # Convert lists and dicts to JSON strings for saving
                            group.attrs[key] = str(value)
                        else:
                            group.attrs[key] = value
            print(f"Results successfully saved to {new_filename}")
        except IOError as e:
            print(f"An error occurred while saving the file: {e}")

    def save_results_to_csv_file(self, results: List[Dict[str, Any]], filename: Union[str, Path]):
        """
        Save the results to an CSV file.

        Args:
            results (List[Dict[str, Any]]): List of results to be saved.
            filename (Union[str, Path]): The base filename or full path for the output file.
        """
        
        # Convert filename to Path object if it is a string
        if isinstance(filename, str):
            filename = Path(path_config.SNR_ANALYZER_OUTPUT_DIR) / filename
        
        # Ensure the directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract the stem and suffix, then append the timestamp
        new_filename = filename.with_name(f"{filename.stem}_{timestamp}.csv")

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(results)
        
        # Save the DataFrame to a CSV file
        df.to_csv(new_filename, index=False)

        print(f"Results successfully saved to {new_filename}")


    def calculate_combined_stats(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate the combined mean, combined standard deviation, and mean range for a grouped DataFrame.

        This function computes the combined statistics for a grouped DataFrame, including:
        - Weighted combined mean: The mean value weighted by the count of observations in each group.
        - Combined standard deviation: The standard deviation that accounts for the variation within each group and the variation between group means.
        - Mean range: The average of the range (max - min) within each group.

        Args:
            group (pd.DataFrame): A DataFrame group (typically resulting from a `groupby` operation) that includes the following columns:
                - 'mean': The mean value for each subgroup.
                - 'std': The standard deviation for each subgroup.
                - 'count': The number of observations in each subgroup.
                - 'min': The minimum value in each subgroup.
                - 'max': The maximum value in each subgroup.

        Returns:
            pd.Series: A Series containing the calculated combined statistics:
                - 'combined_mean': The weighted combined mean across all subgroups.
                - 'combined_std': The combined standard deviation across all subgroups.
                - 'mean_range': The average range (max - min) across all subgroups.

        Example:
            combined_stats = df.groupby('category').apply(calculate_combined_stats)
        """

        # Weighted combined mean
        combined_mean = (group['mean'] * group['count']).sum() / group['count'].sum()
        
        # Differences between means and combined mean
        d = group['mean'] - combined_mean

        # Combined standard deviation
        combined_std = np.sqrt(
            (group['count'] * (group['std']**2 + d**2)).sum() / group['count'].sum()
        )

        # Calculate MAD for each group using the group's own mean
        group_mad = group.apply(lambda row: np.mean(np.abs(row['mean'] - group['mean'])), axis=1)

        # Combined MAD (weighted by group sizes)
        combined_mad = (group_mad * group['count']).sum() / group['count'].sum()


        return pd.Series({
            'combined_mean': combined_mean, 
            'combined_std': combined_std, 
            'combined_mad': combined_mad
        })

    def calculate_snr_scores(self,
                             df: pd.DataFrame,
                             category: str,
                             snr_threshold: float = 0.0,
                             steepness: float = 0.001,
                             alpha: float = 0.5,
                             beta: float = 0.5) -> pd.DataFrame:
        """
        Calculate Signal-to-Noise Ratio (SNR) scores and related statistics for a specified category.

        This function computes the Signal-to-Noise Ratio (SNR) scores for each group within a specified category 
        in the DataFrame. It calculates basic statistics like mean, standard deviation (std), and Mean Absolute 
        Deviation (MAD) for the SNR values. Additionally, a penalty is applied to the SNR score if the mean SNR 
        is below a given threshold, using a logarithmic penalty formula to adjust the score based on the deviation 
        from the threshold.

        The final SNR score reflects both the distribution of the SNR values within the group (through std and MAD) 
        and any penalty for having a mean SNR below the threshold. The function also includes parameters to control 
        the steepness of the penalty and the relative weight of the std and MAD in the final score.

        Args:
            df (pd.DataFrame): The DataFrame containing the SNR data to be analyzed. It must contain at least the 
                            specified `category` column and the `snr` column with numerical SNR values.
            category (str): The column name by which to group the data before calculating the SNR statistics (e.g., 
                            'region', 'subject').
            snr_threshold (float): The minimum acceptable threshold for a "good" SNR score. If the mean SNR of a group 
                                is below this threshold, a logarithmic penalty is applied (default=0.0 dB).
            steepness (float): A parameter controlling the severity of the penalty applied when the mean SNR is below 
                            the threshold. A higher value leads to a stronger penalty for deviations (default=0.001).
            alpha (float): A weighting factor controlling the influence of the standard deviation (std) in the final SNR score.
            beta (float): A weighting factor controlling the influence of the Mean Absolute Deviation (MAD) in the final SNR score.

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
                - category: The category by which the data was grouped.
                - mean: The mean SNR value for each group.
                - std: The standard deviation of the SNR values within each group.
                - count: The number of observations in each group.
                - mad: The Mean Absolute Deviation of the SNR values within each group.
                - penalty_factor: The penalty factor applied to the final SNR score. It is computed based on whether 
                                the mean SNR is below the threshold, and the penalty is logarithmic in nature.
                - snr_score: The final SNR score for each group. It accounts for the variability in SNR values (std, MAD) 
                            and the penalty applied if the mean SNR is below the threshold.

        Example:
            snr_scores = calculate_snr_scores(df, category='region', snr_threshold=0.0, steepness=0.001, alpha=0.5, beta=0.5)

        The function will group the data by the 'region' column, calculate the necessary statistics, apply penalties if 
        the mean SNR is below 0.0 dB, and return a DataFrame with the final SNR scores and other statistical details.
        """

        # First calculate the basic statistics for each category
        snr_scores = df.groupby(category).agg(
            mean=('snr', 'mean'),
            median=('snr', 'median'),
            std=('snr', 'std'),
            count=('snr', 'count')
        ).reset_index()

        # Calculate MAD for each group based on raw 'snr' values
        snr_scores['mad'] = df.groupby(category)['snr'].apply(lambda s: np.mean(np.abs(s - np.mean(s)))).reset_index(drop=True)

        snr_scores = snr_scores.assign(
            penalty_factor=lambda x:
                np.where(
                    x['mean'] < snr_threshold,
                    np.maximum(1 - np.log(1 + steepness * np.abs(x['mean'])), 0.1),
                    1 # No penalty if within range
                ),

            # Final SNR score with penalties applied
            snr_score=lambda x: (
                (1 / (1 + alpha * x['std'] + beta * x['mad'])) * x['penalty_factor']
            )
        )

        # print(snr_scores.head())

        return snr_scores

    def calculate_hr_scores(self,
                            df: pd.DataFrame,
                            category: str,
                            hr_threshold_low: int = 48,
                            hr_threshold_high: int = 150,
                            steepness: float = 0.001,
                            alpha: float = 0.5,
                            beta: float = 0.5) -> pd.DataFrame:
        """
        Calculate heart rate (HR) scores and related statistics for a specified category.

        This function computes various heart rate (HR) statistics for each group within a specified category 
        in the DataFrame. It calculates basic statistics such as the mean, standard deviation, count, minimum, 
        and maximum heart rate values. The function also generates a composite HR score for each category 
        based on these metrics, applying penalties for heart rates that fall outside a specified threshold range.

        Additionally, it computes penalties for HR values outside the defined range (hr_threshold_low to hr_threshold_high) 
        using a logarithmic penalty formula, and then calculates the final HR score, which reflects the variability of 
        HR values (standard deviation and Mean Absolute Deviation, MAD) and the penalties for out-of-range values.

        Args:
            df (pd.DataFrame): The DataFrame containing heart rate data. It must include a `heart_rate` column and 
                            a category column to group the data.
            category (str): The column name by which to group the data (e.g., 'region', 'subject').
            hr_threshold_low (int): The lower bound for the acceptable HR range (default=48 BPM).
            hr_threshold_high (int): The upper bound for the acceptable HR range (default=150 BPM).
            steepness (float): A parameter controlling the severity of the penalty applied when the HR mean is 
                            outside the threshold range (default=0.001).
            alpha (float): Weighting factor for the influence of standard deviation in the final HR score (default=0.5).
            beta (float): Weighting factor for the influence of Mean Absolute Deviation (MAD) in the final HR score (default=0.5).

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
                - category: The grouping category (e.g., 'region', 'subject').
                - combined_mean: The combined mean heart rate across subgroups.
                - combined_std: The combined standard deviation of heart rates across subgroups.
                - combined_mad: The combined Mean Absolute Deviation (MAD) of heart rates across subgroups.
                - penalty_factor: The penalty applied to the HR score based on whether the mean HR is outside the threshold range.
                - hr_score: The final HR score for each category, accounting for variability and penalties.

        Example:
            hr_scores = calculate_hr_scores(df, category='region', hr_threshold_low=48, hr_threshold_high=150, 
                                            steepness=0.001, alpha=0.5, beta=0.5)

            This will calculate HR scores for each 'region', applying penalties for regions where the mean HR falls 
            outside the range of 48-150 BPM, and returns the combined HR statistics and scores.
        """
        
        subfolder_stats = df.groupby([category, 'subfolder'])['heart_rate'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate combined stats for each category
        hr_scores = subfolder_stats.groupby(category, group_keys=False).apply(self.calculate_combined_stats, include_groups=False).reset_index()
        
        hr_scores = hr_scores.assign(
            penalty_factor=lambda x: np.where(
                x['combined_mean'] < hr_threshold_low,
                np.maximum(1 - np.log(1 + steepness * np.abs(hr_threshold_low - x['combined_mean'])), 0.1),
                np.where(
                    x['combined_mean'] > hr_threshold_high,
                    np.maximum(1 - np.log(1 + steepness * np.abs(x['combined_mean'] - hr_threshold_high)), 0.1),
                    1  # No penalty if within range
                )
            ),

            # Final HR score with penalties applied
            hr_score=lambda x: (
                (1 / (1 + alpha * x['combined_std'] + beta * x['combined_mad'])) * x['penalty_factor']
            )
        ).reset_index(drop=True)

        # print(hr_scores.head())

        return hr_scores

    def calculate_combined_in_range_proportion(self,
                                               df: pd.DataFrame,
                                               category: str,
                                               hr_threshold_low: int = 48,
                                               hr_threshold_high: int = 150,
                                               snr_threshold: float = 0.0) -> pd.DataFrame:
        """
        Calculate the proportion of data points that satisfy both HR and SNR conditions for each group.

        This function computes the proportion of data points where both the heart rate (HR) values are within 
        a specified range (between `hr_threshold_low` and `hr_threshold_high`), and the Signal-to-Noise Ratio (SNR) 
        values are above a given threshold (e.g., SNR > `snr_threshold`). The calculation is done for each group 
        specified by the `category` column.

        Args:
            df (pd.DataFrame): The DataFrame containing the HR and SNR data to be analyzed. 
                            It must include at least two columns: `heart_rate` and `snr`.
            category (str): The column by which to group the data (e.g., 'region', 'subject').
            hr_threshold_low (int): The lower bound for the acceptable HR range (default=48 BPM).
            hr_threshold_high (int): The upper bound for the acceptable HR range (default=150 BPM).
            snr_threshold (float): The threshold for the acceptable SNR value (default=0.0 dB).

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
                - category: The grouping category (e.g., 'region', 'subject').
                - in_range_proportion: The proportion of data points that satisfy both HR and SNR conditions for each group.

        Example:
            combined_proportions = calculate_combined_in_range_proportion(df, category='region', 
                                                                        hr_threshold_low=48, 
                                                                        hr_threshold_high=150, 
                                                                        snr_threshold=0.0)
        
            This function will return the proportion of data points in each 'region' where the HR is between 
            48 and 150 BPM, and the SNR is above 0 dB.
        """

        # Calculate individual proportions for HR in range and SNR above threshold
        proportions = df.groupby(category).apply(
            lambda x: pd.Series({
                'hr_in_range_proportion': ((x['heart_rate'] >= hr_threshold_low) & 
                                        (x['heart_rate'] <= hr_threshold_high)).sum() / len(x),
                'snr_above_threshold_proportion': (x['snr'] > snr_threshold).sum() / len(x),
                'combined_in_range_proportion': ((x['heart_rate'] >= hr_threshold_low) & 
                                                (x['heart_rate'] <= hr_threshold_high) & 
                                                (x['snr'] > snr_threshold)).sum() / len(x)
            })
        ).reset_index()

        # print(proportions.head())

        # # Calculate the proportion of values where HR is within range and SNR is above the threshold
        # combined_in_range_proportion = df.groupby(category).apply(
        #     lambda x: ((x['heart_rate'] >= hr_threshold_low) &
        #             (x['heart_rate'] <= hr_threshold_high) &
        #             (x['snr'] > snr_threshold)).sum() / len(x),
        #     include_groups=False
        # ).reset_index(name='in_range_proportion')

        # print(combined_in_range_proportion.head())

        return proportions

    def calculate_combined_scores(self,
                                  df: pd.DataFrame,
                                  category: str,
                                  snr_weight: float = 0.33,
                                  hr_weight: float = 0.33,
                                  frequency_weight: float = 0.33) -> pd.DataFrame:
        """
        Calculate and combine SNR, HR, and frequency in-range scores.

        This function calculates a combined score based on the Signal-to-Noise Ratio (SNR), heart rate (HR),
        and the in-range proportion of both SNR and HR values within specified thresholds. It uses weighted
        averaging to combine the individual scores (SNR, HR, and in-range proportion) into a single composite score.

        The combined score is calculated as a weighted sum of the normalized SNR and HR scores, along with the 
        frequency in-range proportion (i.e., the proportion of values that meet the SNR and HR conditions). 
        Users can adjust the relative importance of each component by providing weights for SNR, HR, and frequency.

        Args:
            df (pd.DataFrame): The DataFrame containing the SNR and HR data to be analyzed.
            category (str): The column name by which to group the data before calculating scores (e.g., 'region', 'subject').
            snr_weight (float): The weight applied to the SNR score in the final combined score (default=0.5).
            hr_weight (float): The weight applied to the HR score in the final combined score (default=0.5).
            frequency_weight (float): The weight applied to the in-range proportion in the final combined score (default=1).

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
                - category: The grouping category (e.g., 'region', 'subject').
                - snr_score: The calculated SNR score for each category.
                - hr_score: The calculated HR score for each category.
                - in_range_proportion: The proportion of values in both HR and SNR ranges.
                - combined_score: The final combined score for each category, based on the weighted sum of the SNR, HR, and in-range scores.

        Example:
            combined_scores = calculate_combined_scores(df, category='region', snr_weight=0.5, hr_weight=0.5, frequency_weight=1)

            This function will calculate the SNR score, HR score, and the combined in-range proportion for each 'region'.
            The final combined score is the weighted sum of these values.
        """

        snr_scores = self.calculate_snr_scores(df, category)
        hr_scores = self.calculate_hr_scores(df, category)
        proportions = self.calculate_combined_in_range_proportion(df, category)

        combined_scores = pd.merge(
            snr_scores[['snr_score', 'mean', 'std', 'mad', category]].reset_index(drop=True),
            hr_scores[['hr_score', 'combined_mean', 'combined_std', 'combined_mad', category]].reset_index(drop=True),
            on=category,
            how='inner'
        )

        combined_scores = pd.merge(
            combined_scores,
            proportions[['combined_in_range_proportion', 'hr_in_range_proportion', 'snr_above_threshold_proportion', category]].reset_index(drop=True),
            on=category,
            how='inner'
        )

        # Calculate the final combined score
        combined_scores['combined_score'] = (
            (combined_scores['snr_score'] * snr_weight) +
            (combined_scores['hr_score'] * hr_weight) +
            (combined_scores['combined_in_range_proportion'] * frequency_weight)
        )

        # print(f"Final Combined Scores:\n{combined_scores.head()}")

        return combined_scores


    def visualize_scatter_plots(self, df: pd.DataFrame, metrics: list, category: str, output_dir: Path, filename_pattern: str):
        """
        Visualize scatter plots for the metrics, grouped by a category.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            metrics (list): A list of metric column names (e.g., ['snr_score', 'hr_score', 'in_range_proportion', 'combined_score']).
            category (str): The categorical column to group by (e.g., 'region', 'video').
            output_dir (Path): Directory where the output images will be saved.
            filename_pattern (str): Pattern for the output filename.
        """
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x=metrics[i], y=metrics[j], hue=category, palette='Set2', s=100, edgecolor='black', alpha=0.7)
                plt.title(f"Scatter Plot of {metrics[i]} vs {metrics[j]} by {category}", fontsize=16)
                plt.xlabel(metrics[i], fontsize=14)
                plt.ylabel(metrics[j], fontsize=14)
                plt.legend(title=category, loc='best', fontsize=12, title_fontsize=14)
                plt.grid(True, linestyle='--', linewidth=0.5)
                plt.tight_layout()

                # Use Path for saving the output file
                output_file = output_dir / f"{filename_pattern}_{metrics[i]}_vs_{metrics[j]}_scatter_by_{category}.png"
                plt.savefig(output_file, dpi=300)
                plt.close()
                print(f"Scatter plot saved to {output_file}")

    def perform_cluster_analysis(self, df: pd.DataFrame, metrics: list, category: str, output_dir: Path, filename_pattern: str, n_clusters: Optional[int] = None):
        """
        Perform K-Means cluster analysis on the metrics and visualize the clusters.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            metrics (list): A list of metric column names (e.g., ['snr_score', 'hr_score', 'in_range_proportion', 'combined_score']).
            category (str): The categorical column to group by (e.g., 'region', 'video').
            output_dir (Path): Directory where the output image will be saved.
            filename_pattern (str): The filename pattern for saving the plot.
            n_clusters (int): The number of clusters to find (default=None). If None, the optimal number of clusters will be chosen dynamically.
        """
        
        # Standardize the metric columns
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[metrics])

        # Dynamically find the optimal number of clusters if not provided
        if n_clusters is None:
            elbow_n_clusters = self.find_optimal_n_clusters(df, metrics, output_dir, filename_pattern) # , best_silhouette_n_clusters
            n_clusters_elbow = elbow_n_clusters

        print(f"Using {n_clusters_elbow} clusters for K-Means (Elbow Method)")
        
        # Perform K-Means clustering for the Elbow Method
        self.cluster_and_plot(df, df_scaled, metrics, category, n_clusters_elbow, output_dir, f"{filename_pattern}_elbow_method", scaler)

    def cluster_and_plot(self, df: pd.DataFrame, df_scaled, metrics: list, category: str,
                         n_clusters: int, output_dir: Path, filename_pattern: str, scaler: StandardScaler, use_minibatch: bool = False):
        """
        Perform K-Means clustering and save the plot for a given number of clusters.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            df_scaled (np.array): Scaled data for clustering.
            metrics (list): A list of metric column names (e.g., ['snr_score', 'hr_score', 'in_range_proportion', 'combined_score']).
            category (str): The categorical column to group by (e.g., 'region', 'video').
            n_clusters (int): The number of clusters to use for K-Means clustering.
            output_dir (Path): Directory where the output image will be saved.
            filename_pattern (str): The filename pattern for saving the plot.
            use_minibatch (bool): If True, use MiniBatchKMeans instead of KMeans for faster execution.
        """
        
        # Perform clustering with the specified number of clusters
        if use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, max_iter=200)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        df['cluster'] = kmeans.fit_predict(df_scaled)

        # Get centroids in scaled space
        centroids_scaled = kmeans.cluster_centers_

        # Reverse the scaling of centroids back to original feature space
        centroids = scaler.inverse_transform(centroids_scaled)

        # Save cluster sizes and centroids to a text file
        cluster_sizes = df['cluster'].value_counts()
        cluster_info_file = output_dir / f"{filename_pattern}_cluster_info.txt"
        
        with open(cluster_info_file, 'w') as f:
            f.write("Cluster Sizes:\n")
            f.write(cluster_sizes.to_string())
            print("\nCluster Sizes:")
            print(cluster_sizes)

            f.write("\n\nCluster Centroids:\n")
            for i, centroid in enumerate(centroids):
                f.write(f"Cluster {i} Centroid: {centroid}\n")
            print("\nCluster Centroids:")
            for i, centroid in enumerate(centroids):
                print(f"Cluster {i} Centroid: {centroid}")
                    
            # Inertia
            inertia = kmeans.inertia_
            f.write(f"\nInertia (Sum of squared distances to closest cluster center): {inertia}\n")
            print(f"\nInertia: {inertia}")

            # Cluster center variances
            cluster_center_variances = np.var(centroids, axis=0)
            f.write(f"Variance of the cluster centroids across features: {cluster_center_variances}\n")
            print(f"\nVariance of cluster centroids across features: {cluster_center_variances}")

            # Save details in CSV for distances to centroids and other metrics
            distances_to_centroids = np.min(kmeans.transform(df_scaled), axis=1)
            df['distance_to_centroid'] = distances_to_centroids
            df[['cluster', 'distance_to_centroid']].to_csv(output_dir / f"{filename_pattern}_distances_to_centroid.csv", index=False)
            f.write(f"\nDistances to centroids saved in CSV.\n")

            # Cluster category distribution
            cluster_category_distribution = df.groupby('cluster')[category].value_counts().unstack(fill_value=0)
            cluster_category_distribution.to_csv(output_dir / f"{filename_pattern}_cluster_category_distribution.csv")
            f.write(f"Cluster category distribution saved to CSV.\n")

            # Within-cluster variances (for numeric columns only)
            numeric_columns = df.select_dtypes(include=np.number)
            within_cluster_variances = numeric_columns.groupby(df['cluster']).var()
            within_cluster_variances.to_csv(output_dir / f"{filename_pattern}_within_cluster_variances.csv")
            f.write(f"Within-cluster variances (numeric columns) saved to CSV.\n")
            
        print(f"Cluster information saved to {cluster_info_file}")

        # Create a color palette for clusters
        palette = sns.color_palette('viridis', n_clusters)

        ### Part 1: Pairplot ###
        pairplot = sns.pairplot(df, vars=metrics, hue='cluster', palette=palette, diag_kind='kde',
                                plot_kws={'alpha': 0.5, 's': 1})  # Smaller points, higher transparency

        # Adjust the layout and the figure size
        pairplot.fig.set_size_inches(10, 6)  # Smaller figure size
        pairplot.fig.suptitle(f"Pairplot Analysis of {category} (n_clusters={n_clusters})", y=1.02, fontsize=12, fontweight='bold')

        # Adjust the legend with larger markers
        handles, labels = pairplot.fig.gca().get_legend_handles_labels()
        pairplot.fig.legend(handles=handles, labels=labels, loc='center right', title="Cluster", markerscale=3.0)

        # Save the pairplot
        pairplot_file = output_dir / f"{filename_pattern}_{category}_pairplot_n{n_clusters}.png"
        pairplot.savefig(pairplot_file, dpi=300, bbox_inches='tight')
        plt.close(pairplot.fig)

        print(f"Pairplot saved to {pairplot_file}")

        # ### Part 2: KDE Plots Side by Side ###
        # plt.figure(figsize=(15, 6))  # Adjust figure size as needed
        # for i, metric in enumerate(metrics):
        #     plt.subplot(1, len(metrics), i + 1)  # Create a subplot for each KDE
        #     # sns.kdeplot(df[metric], hue='cluster', palette=palette, fill=True, alpha=0.3)

        #     for j in range(n_clusters):
        #         cluster_data = df[df['cluster'] == j]  # Subset data for each cluster
        #         sns.kdeplot(cluster_data[metric], label=f'Cluster {j}', hue='cluster', palette=palette[j], fill=True, alpha=0.5)
                
        #     plt.title(f'{metric} KDE Plot', fontsize=12, fontweight='bold')
        #     plt.xlabel(metric, fontsize=11)
        #     plt.ylabel('Density', fontsize=11)

        # # Add a common title for the entire figure
        # plt.suptitle(f"KDE Plot Analysis of {category} (n_clusters={n_clusters})", fontsize=12, fontweight='bold')

        # # Save the KDE plot
        # kde_file = output_dir / f"{filename_pattern}_{category}_kdeplot_n{n_clusters}.png"
        # plt.tight_layout(rect=(0, 0, 1, 0.95))  # Adjust layout to fit title
        # plt.savefig(kde_file, dpi=300, bbox_inches='tight')
        # plt.close()

        # print(f"KDE plots saved to {kde_file}")

        ### Part 3: Scatter Plot with Convex Hulls ###
        plt.figure(figsize=(10, 6))

        # Scatter plot for the first two metrics (assuming these are snr and heart_rate)
        sns.scatterplot(x=df[metrics[0]], y=df[metrics[1]], hue=df['cluster'], palette=palette, alpha=0.5, s=10)

        # Add convex hulls to the scatter plot
        for i in range(n_clusters):
            cluster_points = df[df['cluster'] == i][[metrics[0], metrics[1]]].values
            self.plot_convex_hull(plt.gca(), cluster_points, palette[i])  # Assuming self.plot_convex_hull is defined

        # Set plot title and labels
        plt.title(f"Scatter Plot with Convex Hulls for {category} (n_clusters={n_clusters})", fontsize=12, fontweight='bold')
        plt.xlabel(metrics[0], fontsize=11)
        plt.ylabel(metrics[1], fontsize=11)
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., fontsize=10)

        # Save the scatter plot with convex hulls
        scatter_file = output_dir / f"{filename_pattern}_{category}_scatter_hull_n{n_clusters}.png"
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot with convex hulls saved to {scatter_file}")

        ### Part 4: Voronoi Diagram ###
        voronoi_file = output_dir / f"{filename_pattern}_voronoi_diagram_n{n_clusters}.png"
        
        # Plot the Voronoi diagram based on the first two features (metrics[0] and metrics[1])
        self.plot_voronoi_diagram(centroids[:, :2], voronoi_file)
        
        # Analyze cluster category distribution
        self.analyze_cluster_category_distribution(df, category, output_dir, filename_pattern)

    def plot_voronoi_diagram(self, centroids, output_file):
        vor = Voronoi(centroids)
        
        # Create a figure for Voronoi diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=2, line_alpha=0.6)
        
        # Add title and labels
        ax.set_title("Voronoi Diagram of Cluster Centroids", fontsize=12, fontweight='bold')
        ax.set_xlabel("SNR")
        ax.set_ylabel("Heart Rate")
        
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Voronoi diagram saved to {output_file}")

    def plot_convex_hull(self, ax, points, color):
        hull = ConvexHull(points)
        # Draw polygon around the convex hull points
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], color=color, lw=2)

    def darken_color(self, color, factor=0.5):
        """
        Darken a given color by a specific factor.
        
        Args:
            color (str): The original color.
            factor (float): Factor by which to darken the color (0 to 1).
        
        Returns:
            tuple: Darkened RGB color.
        """
        c = mcolors.to_rgb(color)  # Convert to RGB
        return (c[0] * factor, c[1] * factor, c[2] * factor)  # Return the darkened color

    def analyze_cluster_category_distribution(self, df: pd.DataFrame, category: str, output_dir: Path, filename_pattern: str):
        """
        Analyze the distribution of categories within clusters and visualize the results.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the cluster and category data.
            category (str): The column containing the category information.
            output_dir (Path): The directory where the output images will be saved.
            filename_pattern (str): The filename pattern for saving the output image.
        """
        # Create a crosstab to analyze the distribution of categories within clusters
        category_cluster_distribution = pd.crosstab(df['cluster'], df[category])
        
        # Print the raw counts of categories within clusters
        print("Frequency of categories within clusters:\n", category_cluster_distribution)
        
        # Calculate the percentage distribution of categories in each cluster
        category_cluster_percentage = category_cluster_distribution.div(category_cluster_distribution.sum(axis=1), axis=0)
        print("\nPercentage distribution of categories within clusters:\n", category_cluster_percentage)
        
        # Plot the distribution as a stacked bar chart
        category_cluster_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Category Distribution by Cluster', fontsize=12, fontweight='bold')
        plt.xlabel('Cluster', fontsize=11)
        plt.ylabel('Count', fontsize=11)
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()

        # Save the plot as a PNG image
        output_file = output_dir / f"{filename_pattern}_category_distribution_by_cluster.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Category distribution plot saved to {output_file}")

    def find_optimal_n_clusters(self, df: pd.DataFrame, metrics: list, output_dir: Path, filename_pattern: str, max_clusters: int = 10):
        """
        Use the Elbow Method and Silhouette Score to find the optimal number of clusters.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            metrics (list): A list of metric column names to be used for clustering.
            output_dir (Path): Directory to save the elbow and silhouette plots.
            filename_pattern (str): The filename pattern for saving the plot.
            max_clusters (int): The maximum number of clusters to test.

        Returns:
            elbow_n_clusters (int): The optimal number of clusters from the Elbow method.
            best_silhouette_n_clusters (int): The optimal number of clusters from the Silhouette method.
        """
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[metrics])

        inertia = []
        max_iter = 200

        # Calculate inertia (within-cluster sum of squared errors) for each number of clusters
        for n_clusters in tqdm(range(2, max_clusters + 1), desc="Calculating K-Means for different cluster sizes", position=1):
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, max_iter=max_iter, init='k-means++', verbose=0)

            # Initialize tqdm progress bar for max_iter batches in MiniBatchKMeans
            for _ in tqdm(range(max_iter), desc=f"K-Means n_clusters={n_clusters}", position=2, leave=False):
                kmeans.partial_fit(df_scaled)

            # Calculate inertia score
            inertia_value = kmeans.inertia_
            print(f"Inertia for {n_clusters} clusters: {inertia_value}")

            # Append the results to lists
            inertia.append(inertia_value)

        print("K-Means clustering completed.")

        # Detect the elbow point using the "knee" detection method from the kneed library
        knee_locator = KneeLocator(range(2, max_clusters + 1), inertia, curve="convex", direction="decreasing")
        elbow_n_clusters = knee_locator.elbow
        print(f"Elbow detected at: {elbow_n_clusters} clusters")

        # Elbow Method: Look for the "elbow" in the inertia plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(2, max_clusters + 1), inertia, marker='o', color='darkgray', linestyle='--', label="Inertia", zorder=1)

        # Highlight the elbow point on the graph
        plt.scatter(elbow_n_clusters, inertia[elbow_n_clusters - 2], color='red', s=100, label='Elbow Point', zorder=2)  # Adjust index

        plt.title("Elbow Method: Inertia vs. Number of Clusters", fontsize=12, fontweight='bold')
        plt.xlabel("Number of Clusters", fontsize=11)
        plt.ylabel("Inertia", fontsize=11)
        plt.grid(True)
        
        plt.legend()

        # Save the plot
        elbow_file = output_dir / f"{filename_pattern}_elbow_method.png"
        plt.savefig(elbow_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Elbow method plot saved to {elbow_file}")

        # Return the number of clusters at the elbow
        return elbow_n_clusters

    def detect_outliers(self, df: pd.DataFrame, metrics: List[str], category: str, output_dir: Path, filename_pattern: str, subsample_frac: float = 0.5):
        """
        Detect outliers in a given metric using the IQR method and save the plots as PNG images.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            metrics (list): A list of metrics to check for outliers.
            category (str): The column for categorical grouping.
            output_dir (Path): Directory where the output image will be saved.
            filename_pattern (str): The filename pattern for saving the plot.
            subsample_frac (float): Fraction of data to subsample for plotting (default=0.5).
        """

        for metric in metrics:
            # Create a figure for Matplotlib plotting
            plt.figure(figsize=(14, 10))
            
            # Initialize a new DataFrame to store outliers
            outliers_df = pd.DataFrame()
            
            # Loop through each unique category and calculate outliers
            for cat in df[category].unique():
                cat_data = df[df[category] == cat][metric]
                
                # Calculate IQR and detect outliers
                Q1 = cat_data.quantile(0.25)
                Q3 = cat_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_whisker = Q1 - 1.5 * IQR
                upper_whisker = Q3 + 1.5 * IQR
            
                # Detect outliers for this category
                cat_outliers = df[(df[category] == cat) & ((cat_data < lower_whisker) | (cat_data > upper_whisker))]
                outliers_df = pd.concat([outliers_df, cat_outliers])  # Combine outliers from each category
                
                # Log the whisker values for verification
                # print(f"{metric} Outlier Detection for {cat} -> Lower Whisker: {lower_whisker}, Upper Whisker: {upper_whisker}")

            # Plot a boxplot with Seaborn
            sns.boxplot(x=category, y=metric, data=df, hue=category, palette="viridis", fliersize=0, width=0.6)
            
            # Overlay the detected outliers as individual points
            sns.stripplot(x=category, y=metric, data=outliers_df, hue=category, palette="viridis", marker="o", size=3)

            # Add title and labels
            plt.title(f'Outlier Detection in {metric} by {category.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            plt.xlabel(category.replace("_", " ").title(), fontsize=11)
            plt.ylabel(metric.replace("_", " ").title(), fontsize=11)
            plt.xticks(rotation=45, ha='right')

            # Tighten the layout for better aesthetics
            plt.tight_layout()
            
            # Save the plot as PNG
            output_file = output_dir / f"{filename_pattern}_{metric}_outliers.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"PNG outliers plot saved to {output_file}")

    def visualize_variability(self, df: pd.DataFrame,
                              snr_col: str, hr_col: str, in_range_col: str, combined_col: str,
                              category_col: str, output_dir: Path, filename_pattern: str):
        """
        Visualize the variability in SNR, HR, and adjusted_in_range_proportion by plotting histograms with KDE for each metric.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the SNR, HR, and adjusted_in_range_proportion data to be analyzed.
            snr_col (str): The column name representing SNR values in the DataFrame.
            hr_col (str): The column name representing HR values in the DataFrame.
            in_range_col (str): The column name representing adjusted_in_range_proportion in the DataFrame.
            combined_col (str): The column name representing Combined Score in the DataFrame.
            category_col (str): The categorical column for hue.
            output_dir (Path): The directory where the plots will be saved.
            filename_pattern (str): The filename pattern for saving the output image.
        """

        plt.figure(figsize=(16, 18))

        # Font settings for titles, labels, and ticks
        title_fontsize = 12
        label_fontsize = 11
        tick_fontsize = 11
        legend_fontsize = 11

        # Set a title above all the plots
        plt.suptitle(f'Histograms and KDE Plots for SNR, HR, In-Range Proportion, and Combined Score by {category_col.replace("_", " ").title()}', 
                    fontsize=title_fontsize, fontweight='bold')
    
        # Individual histograms for each metric with the customized colors
        metrics = [
            (snr_col, 'SNR'),
            (hr_col, 'HR'),
            (in_range_col, 'In-Range Proportion'),
            (combined_col, 'Combined Score')
        ]

        for i, (col, label) in enumerate(metrics):
            # Histogram with KDE for each metric
            plt.subplot(4, 1, i + 1)
            hist = sns.histplot(df, x=col, hue=category_col, kde=True, palette='viridis', bins=20,
                         element="step", stat="density", common_norm=False)
            plt.title(f'Histogram and KDE of {label} by {category_col}')
            plt.xlabel(label, fontsize=label_fontsize)
            plt.ylabel('Density', fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)

            # Hide redundant legends for subsequent plots
            if i > 0:
                hist.legend_.remove()
            
        plt.subplots_adjust(hspace=0.4)
        plt.tight_layout()

        output_file = output_dir / f"{filename_pattern}_scores_histograms_variability.png"
        plt.savefig(output_file)
        plt.close()

        print(f"Histograms with KDE plot saved to {output_file}")

    def visualize_variability_impact(self, df: pd.DataFrame,
                                     snr_col: str, hr_col: str, in_range_col: str, combined_col: str,
                                     category_col: str, output_dir: Path, filename_pattern: str):
        """
        Visualize the impact of variability in SNR, HR, and adjusted_in_range_proportion on the Combined Score,
        with the ability to distinguish different categories using hue.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the SNR, HR, adjusted_in_range_proportion, and Combined Score data to be analyzed.
            snr_col (str): The column name representing SNR values in the DataFrame.
            hr_col (str): The column name representing HR values in the DataFrame.
            in_range_col (str): The column name representing adjusted_in_range_proportion in the DataFrame.
            combined_col (str): The column name representing Combined Score in the DataFrame.
            category_col (str): The categorical column used for the hue.
            output_dir (Path): The directory where the plots will be saved.
            filename_pattern (str): The filename pattern for saving the output image.
        """

        # Font settings for titles, labels, and ticks
        title_fontsize = 12
        label_fontsize = 11
        tick_fontsize = 11
        legend_fontsize = 11

        plt.figure(figsize=(15, 12))

        # Add a title above all plots
        plt.suptitle('Scatter Plots of SNR, HR, and In-Range Proportion vs Combined Score', fontsize=title_fontsize, fontweight='bold')

        # Color palette for categories
        palette = sns.color_palette('viridis', n_colors=len(df[category_col].unique()))

        # Individual scatter plots for each metric with the customized palette
        metrics = [
            (snr_col, 'SNR'),
            (hr_col, 'HR'),
            (in_range_col, 'In-Range Proportion'),
        ]

        for i, (col, label) in enumerate(metrics):
            plt.subplot(3, 1, i + 1)
            scatter = sns.scatterplot(data=df, x=col, y=combined_col, hue=category_col, palette=palette,
                            alpha=0.8, edgecolor='black', s=100)
            plt.title(f'{label} vs Combined Score', fontsize=title_fontsize)
            plt.xlabel(label, fontsize=label_fontsize)
            plt.ylabel('Combined Score', fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)

            # Hide redundant legends for subsequent plots
            if i > 0:
                scatter.legend_.remove()
            else:
                plt.legend(title=category_col.replace("_", " ").title(), fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        
        plt.subplots_adjust(hspace=0.4)
        plt.tight_layout()

        output_file = output_dir / f"{filename_pattern}_scatter_variability_impact.png"
        plt.savefig(output_file)
        plt.close()

        print(f"Scatter plot for variability impact saved to {output_file}")


    def bar_plot_scores(self, df: pd.DataFrame, category_col: str, snr_col: str, hr_col: str, adjusted_in_range_proportion_col: str, combined_col: str, output_dir: Path, filename_pattern: str):
        # Sort by category (if it's time series) or another feature
        df_sorted = df.sort_values(by=category_col)

        # Format category_col by removing underscores and capitalizing each word
        formatted_category_col = category_col.replace('_', ' ').title()

        # Set up the figure
        plt.figure(figsize=(12, 8))

        # Set the width of each bar
        bar_width = 0.2
        categories = df_sorted[category_col].unique()
        x = np.arange(len(categories))

        # Use a color palette with better contrast
        colors = {
            'SNR': sns.color_palette("Greens", 7)[2],
            'HR': sns.color_palette("Blues", 7)[3],
            'InRange': sns.color_palette("Purples", 7)[4],
            'Combined': sns.color_palette("Reds", 7)[5]
        }

        # Create bars for each score with improved formatting (edgecolor for better visibility)
        plt.bar(x - 1.5*bar_width, df_sorted[snr_col], width=bar_width, label='SNR Score', color=colors['SNR'], edgecolor='black')
        plt.bar(x - 0.5*bar_width, df_sorted[hr_col], width=bar_width, label='HR Score', color=colors['HR'], edgecolor='black')
        plt.bar(x + 0.5*bar_width, df_sorted[adjusted_in_range_proportion_col], width=bar_width, label='In-HR-Range Proportion', color=colors['InRange'], edgecolor='black')
        plt.bar(x + 1.5*bar_width, df_sorted[combined_col], width=bar_width, label='Combined Score', color=colors['Combined'], edgecolor='black')


        # Add labels and title
        plt.xlabel(formatted_category_col, fontsize=14)
        plt.ylabel("Scores", fontsize=14)
        plt.title("Comparison of SNR, HR, In-HR-Range Proportion, and Combined Scores", fontsize=16, fontweight='bold')

        # Format the category labels on the x-axis
        formatted_categories = [category.replace('_', ' ').title() for category in df_sorted[category_col]]
        plt.xticks(x, formatted_categories, rotation=45, ha="right", fontsize=12)

        # Add gridlines for readability
        plt.grid(True, axis='y', linestyle='--', linewidth=0.7)

        # Set the legend outside of the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)

        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_pattern}_scores_barplot.png")
        plt.close()

    def heatmap_correlation(self, df: pd.DataFrame, snr_col: str, hr_col: str, adjusted_in_range_proportion_col: str, combined_col: str, output_dir: Path, filename_pattern: str):
        # Calculate the correlation matrix
        corr = df[[snr_col, hr_col, adjusted_in_range_proportion_col, combined_col]].corr()
        
        # Plot the heatmap
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(corr, annot=True, cmap="RdBu_r", linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 11})
        
        # Add gridlines for better cell separation
        plt.grid(True, which='minor', color='white', linestyle='-', linewidth=1)

        plt.title("Correlation Between SNR Score, HR Score, In-Range Proportion, and Combined Score", fontsize=12, fontweight='bold')
        
        # Improve the axis labels
        ax.set_xticklabels(['SNR Score', 'HR Score', 'In-Range Proportion', 'Combined Score'], fontsize=11, rotation=45, ha="right")
        ax.set_yticklabels(['SNR Score', 'HR Score', 'In-Range Proportion', 'Combined Score'], fontsize=11, rotation=45)
        
        # Add axis titles
        ax.set_xlabel("Metrics", fontsize=11)
        ax.set_ylabel("Metrics", fontsize=11)

        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_pattern}_scores_correlation_heatmap.png")
        plt.close()

    def bar_plot_combined_scores(self, df: pd.DataFrame, category_col: str, combined_col: str, output_dir: Path, filename_pattern: str, top_n: int):
        # Sort DataFrame by combined score and take top N
        df_sorted = df.sort_values(by=combined_col, ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        bar_plot = sns.barplot(x=category_col, y=combined_col, data=df_sorted, hue=category_col, palette='coolwarm_r', edgecolor='black', dodge=False, legend=False)
        
        # Title and labels
        plt.title(f"Top {top_n} Categories by Combined Score", fontsize=16, fontweight='bold')
        plt.ylabel("Combined Score", fontsize=14)
        plt.xlabel(category_col.replace('_', ' ').title(), fontsize=14)

        # Format x-axis labels: remove underscores and capitalize each word
        formatted_labels = [label.replace('_', ' ').title() for label in df_sorted[category_col]]
        
        # Set the x-ticks and corresponding labels
        bar_plot.set_xticks(range(len(df_sorted[category_col])))
        bar_plot.set_xticklabels(formatted_labels, rotation=45, ha="right", fontsize=12)
        
        # Add gridlines for better readability
        bar_plot.grid(True, axis='y', linestyle='--', linewidth=0.7)

        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_pattern}_combined_scores_barplot.png")
        plt.close()

    def visualize_facet_grid(self, df: pd.DataFrame, metric_col: str, category_col: str, output_dir: Path, filename_pattern: str):
        sns.set(style="whitegrid")

        # Format metric_col and category_col
        formatted_metric_col = 'Heart Rate (bpm)' if metric_col == 'heart_rate' else 'SNR (dB)'
        formatted_category_col = category_col.replace('_', ', ').title()
        plot_title = f"{formatted_metric_col} Distribution by {formatted_category_col}"
        
        # Get the number of unique categories
        num_categories = df[category_col].nunique()

        # Dynamically adjust the grid layout based on the number of categories
        col_wrap = min(3, num_categories)  # If categories are less than 3, set col_wrap to num_categories
        num_rows = (num_categories // col_wrap) + int(num_categories % col_wrap > 0)  # Calculate the number of rows needed

        # Set the height and aspect dynamically based on the number of rows
        height = 4  # Base height per plot
        aspect = 1.5  # Base aspect ratio per plot

        # Create the FacetGrid with the dynamic row and column setup, and hue on category_col
        g = sns.FacetGrid(df, col=category_col, col_wrap=col_wrap, height=height, aspect=aspect,
                          despine=True, margin_titles=True, palette="viridis")

        # Plot KDE plot with a fill
        g.map(sns.kdeplot, metric_col, fill=True, lw=2, warn_singular=False)

        # Set axis limits and highlight color ranges based on the metric
        for ax, category in zip(g.axes.flat, df[category_col].unique()):
            cat_data = df[df[category_col] == category]
            
            if metric_col == 'heart_rate':
                # Set x-axis limits and color range for heart rate (48-150)
                lower_bound, upper_bound = 48, 150
                g.set(xlim=(df[metric_col].min(), df[metric_col].max()))

                # Calculate percentage of values in range for this category
                percentage_in_range = (cat_data[(cat_data[metric_col] >= lower_bound) & (cat_data[metric_col] <= upper_bound)].shape[0] / cat_data.shape[0]) * 100
                print(f"Category: {category}, Percentage of heart_rate in range [48, 150]: {percentage_in_range:.2f}%")

                ax.axvspan(lower_bound, upper_bound, color='lightgreen', alpha=0.3)
                ax.text(0.75, 0.8, f"In range \n(48 - 150 bpm): \n{percentage_in_range:.2f}%", transform=ax.transAxes, fontsize=11, color='black')

                # Find the peak and annotate it
                x_data = cat_data[metric_col].values
                kde = sns.kdeplot(x_data, ax=ax)
                kde_data = kde.get_lines()[0].get_data() # Get KDE data: (x-values, y-values)
                max_density_idx = kde_data[1].argmax() # Find the index of the maximum density

                peak_x = kde_data[0][max_density_idx] # x-coordinate of the peak
                peak_y = kde_data[1][max_density_idx] # y-coordinate of the peak
                ax.annotate(f'Peak: {peak_x:.2f}', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.01),
                            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, color='black')

            elif metric_col == 'snr':
                # Set x-axis limits and color range for snr (values > 0)
                lower_bound = 0
                g.set(xlim=(df[metric_col].min(), df[metric_col].max()))

                # Calculate percentage of values greater than 0 for this category
                percentage_above_zero = (cat_data[cat_data[metric_col] > 0].shape[0] / cat_data.shape[0]) * 100
                print(f"Category: {category}, Percentage of snr values > 0: {percentage_above_zero:.2f}%")

                ax.axvspan(lower_bound, cat_data[metric_col].max(), color='lightblue', alpha=0.3)
                ax.text(0.75, 0.8, f"Above 0 dB: \n{percentage_above_zero:.2f}%", transform=ax.transAxes, fontsize=11, color='black')

                # Find the peak and annotate it
                x_data = cat_data[metric_col].values
                kde = sns.kdeplot(x_data, ax=ax)
                kde_data = kde.get_lines()[0].get_data() # Get KDE data: (x-values, y-values)
                max_density_idx = kde_data[1].argmax() # Find the index of the maximum density

                peak_x = kde_data[0][max_density_idx] # x-coordinate of the peak
                peak_y = kde_data[1][max_density_idx] # y-coordinate of the peak
                ax.annotate(f'Peak: {peak_x:.2f}', xy=(peak_x, peak_y), xytext=(peak_x, peak_y + 0.01),
                            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, color='black')

        # Set titles and labels with improved font size
        g.set_axis_labels(f"{formatted_metric_col}", "Density", fontsize=11)
        g.set_titles("{col_name}", size=12)
        
        for ax in g.axes.flat:
            title = ax.get_title()
            formatted_title = title.replace('_', ' ').title()
            ax.set_title(formatted_title, size=12)
        
        # Add gridlines for better readability
        for ax in g.axes.flat:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust layout and add a global title with improved font size
        g.fig.subplots_adjust(top=0.85, hspace=0.4)
        g.fig.suptitle(plot_title, fontsize=12, fontweight='bold')

        # Save the facet grid
        g.savefig(output_dir / f"{filename_pattern}_{metric_col.lower().replace(' ', '_')}_facet_grid.png")
        plt.close()
    
    
    def visualize_post_hoc_results(self, post_hoc_results: pd.DataFrame, metric: str, category: str, output_dir: Path, filename_pattern: str) -> None:
        """
        Visualize the results of Dunn's post-hoc test as a heatmap.
        
        Args:
            post_hoc_results (pd.DataFrame): The pairwise comparison p-values from Dunn's post-hoc test.
            metric (str): The metric being analyzed (e.g., 'snr' or 'heart_rate').
            category (str): The category used for grouping (e.g., 'roi_type', 'region').
            output_dir (str): The directory where the visualization will be saved.
            filename_pattern (str): The filename pattern to save the heatmap.
        """
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(post_hoc_results, annot=True, cmap='coolwarm', cbar_kws={'label': 'p-value'}, vmin=0, vmax=1, annot_kws={'fontsize': 8})
        
        # Set plot title and labels
        plt.title(f"Dunn's Post-hoc Test Results for {metric.replace('_', ' ').title()}", fontsize=12, fontweight='bold')

        # Set axis labels
        plt.xlabel(f"{category.replace('_', ' ').title()}")
        plt.ylabel(f"{category.replace('_', ' ').title()}")

        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)

        # Save the plot
        output_path = output_dir / f"{filename_pattern}_{metric}_dunn_posthoc.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Post-hoc heatmap saved as {output_path}")

    def calculate_and_visualize_correlations(self, df: pd.DataFrame, output_dir: Path, filename_pattern: str):
        """
        Calculate and visualize the correlation matrix for different metrics, including SNR, HR, Combined Score,
        and Adjusted In-Range Proportion.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the relevant metrics for analysis.
            output_dir (Path): The directory where the correlation heatmap will be saved.
            filename_pattern (str): The filename pattern for saving the heatmap image.
        """
        # Calculate the correlation matrix
        corr_matrix = df.corr()

        # Create the figure for plotting the heatmap
        plt.figure(figsize=(10, 8))
        
        # Plot the correlation matrix using Seaborn's heatmap
        ax = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5,
                         cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 11})

        # Add gridlines for better cell separation
        plt.grid(True, which='minor', color='white', linestyle='-', linewidth=1)

        # Set the title for the heatmap
        plt.title("Correlation Matrix: SNR (dB), HR (bpm), In-Range Proportion, and Combined Score", fontsize=12, fontweight='bold')
        
        # Set the x-axis and y-axis labels
        ax.set_xticklabels(['SNR', 'HR', 'In-Range Proportion', 'Combined Score'], fontsize=11, rotation=45, ha="right")
        ax.set_yticklabels(['SNR', 'HR', 'In-Range Proportion', 'Combined Score'], fontsize=11, rotation=45)
        
        # Set axis labels
        ax.set_xlabel("Metrics", fontsize=11)
        ax.set_ylabel("Metrics", fontsize=11)

        # Save the heatmap image to the specified directory
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_pattern}_snr_hr_combined_inrange_correlation_heatmap.png", bbox_inches='tight')
        plt.close()

    def calculate_and_visualize_snr_hr_correlation(self, df: pd.DataFrame, output_dir: Path, filename_pattern: str):
        """
        Calculate and visualize the correlation matrix for SNR and HR.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the SNR and HR metrics for analysis.
            output_dir (Path): The directory where the heatmap will be saved.
            filename_pattern (str): The filename pattern for saving the heatmap image.
        """
        # Calculate the correlation matrix
        corr_matrix = df.corr()

        # Create the figure for plotting the heatmap
        plt.figure(figsize=(6, 6))
        
        # Plot the correlation matrix using Seaborn's heatmap
        ax = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5,
                         cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 11})

        # Add gridlines for better cell separation
        plt.grid(True, which='minor', color='white', linestyle='-', linewidth=1)

        # Set the title for the heatmap
        plt.title("Correlation Between SNR (dB) and HR (bpm)", fontsize=12, fontweight='bold')

        # Set the x-axis and y-axis labels
        ax.set_xticklabels(['SNR', 'HR'], fontsize=11, rotation=45, ha="right")
        ax.set_yticklabels(['SNR', 'HR'], fontsize=11, rotation=45)
        
        # Set axis labels
        ax.set_xlabel("Metrics", fontsize=11)
        ax.set_ylabel("Metrics", fontsize=11)

        # Save the heatmap image to the specified directory
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_pattern}_snr_hr_correlation_heatmap.png", bbox_inches='tight')
        plt.close()

    def calculate_metrics_correlation_with_category(self, df: pd.DataFrame, category: str, output_dir: Path, filename_pattern: str):
        """
        Calculate and visualize the correlation matrix for different metrics, including SNR, HR, Combined Score,
        and Adjusted In-Range Proportion. The categorical column is label encoded for correlation analysis.

        Args:
            df (pd.DataFrame): The DataFrame containing the relevant metrics for analysis.
            category (str): The categorical column to be included in the correlation matrix.
            output_dir (Path): The directory where the correlation heatmap will be saved.
            filename_pattern (str): The filename pattern for saving the heatmap image.
        """
        # Encode the categorical column
        label_encoder = LabelEncoder()
        df.loc[:, category + '_encoded'] = label_encoder.fit_transform(df[category])

        # Select only the necessary columns for correlation analysis
        relevant_columns = ['snr', 'heart_rate', category + '_encoded']
        df_filtered = df[relevant_columns].copy()

        # Calculate the correlation matrix
        corr_matrix = df_filtered.corr()

        # Create the figure for plotting the heatmap
        plt.figure(figsize=(10, 8))
        
        # Plot the correlation matrix using Seaborn's heatmap
        ax = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5,
                         cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 11})

        # Add gridlines for better cell separation
        plt.grid(True, which='minor', color='white', linestyle='-', linewidth=1)

        # Set the title for the heatmap
        plt.title(f"Correlation Between SNR (dB), HR (bpm), and {category.replace('_', ' ').title()}", fontsize=12, fontweight='bold')

        # Set the x-axis and y-axis labels based on the actual columns in the correlation matrix
        labels = ['SNR', 'HR', category.replace('_', ' ').title()]
        ax.set_xticklabels(labels, fontsize=11, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=11, rotation=45)

        # Set axis labels
        ax.set_xlabel("Metrics", fontsize=11)
        ax.set_ylabel("Metrics", fontsize=11)

        # Save the heatmap image to the specified directory
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_pattern}_snr_hr_{category}_correlation_heatmap.png", bbox_inches='tight')
        plt.close()

        print(f"Correlation heatmap saved to {output_dir}/{filename_pattern}_snr_hr_{category}_correlation_heatmap.png")

    def visualize_test_results(self, results: dict, metrics: list, output_dir: Path, filename_pattern: str,  alpha: float = 0.05):
        # Extracting p-values and tests used for each metric
        p_values = [results[metric]['p_value'] for metric in metrics]
        tests_used = [results[metric]['test_used'] for metric in metrics]
        normality = [results[metric]['normality'] for metric in metrics]
        homoscedasticity = [results[metric]['homoscedasticity'] for metric in metrics]
        
        # 1. Scatter Plot of p-values
        self.visualize_test_results_scatterplot(metrics, p_values, output_dir, filename_pattern, alpha)

        # 2. Heatmap for Normality, Homoscedasticity, and Test Used
        self.visualize_test_results_heatmap(metrics, tests_used, normality, homoscedasticity, output_dir, filename_pattern)

    def visualize_test_results_scatterplot(self, metrics: List[str], p_values: List[float], output_dir: Path, filename_pattern: str, alpha: float = 0.05):
        plt.figure(figsize=(10, 6))
        
        # Generate numerical x-coordinates for metrics
        x_coords = range(len(metrics))

        # Use RdBu_r colors for scatter points
        color = 'darkgray'

        # Adjust point size based on p-values
        point_sizes = [max(50, 200 * (1 - p_val)) for p_val in p_values]  # scale for visibility

        # Create the scatter plot
        scatter = plt.scatter(
            x_coords, p_values, color=color, s=point_sizes, edgecolor='black'
        )

        # Add horizontal line for the significance threshold
        plt.axhline(y=alpha, color='red', linestyle='--', label=f'Significance Threshold (alpha={alpha})')
        
        # Annotate each point with the p-value
        for i, txt in enumerate(p_values):
            plt.annotate(f"{txt:.4g}", (x_coords[i], p_values[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=11)
        
        # Adjust y-axis limits for better visibility
        plt.ylim([min(p_values) - 0.02, max(p_values) + 0.06])

        # Title and axis labels
        plt.title('P-values for Statistical Significance Tests', fontsize=12, fontweight='bold')
        plt.ylabel('P-value', fontsize=11)
        plt.xlabel('Metrics', fontsize=11)
        
        # Set x-ticks with metric names
        plt.xticks(ticks=x_coords, labels=metrics, rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)

        plt.legend(fontsize=11)

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename_pattern}_p_values_scatterplot.png')
        plt.close()

    def visualize_test_results_heatmap(self, metrics: List[str], tests: List[str], normality: List[bool], homoscedasticity: List[bool], output_dir: Path, filename_pattern: str):
        """
        Visualize test results using a heatmap, with a legend displayed outside the plot.
        Ensures consistent colors between the heatmap and the legend.
        """
        # Create DataFrame from test results
        test_results = pd.DataFrame({
            'Test': tests,
            'Normality': normality,
            'Homoscedasticity': homoscedasticity
        }, index=metrics)

        # Binary version for coloring: 1 for True (pass), 0 for False (fail)
        binary_test_results = test_results[['Normality', 'Homoscedasticity']].applymap(lambda x: 1 if x else 0)

        # Define a custom color palette (green for pass, red for fail)
        cmap = sns.color_palette(['#4CAF50', '#FF6F61']) # Red for False, Green for True

        # Set up the figure
        plt.figure(figsize=(10, 6))

        # Create the heatmap for Normality and Homoscedasticity
        ax = sns.heatmap(binary_test_results, cmap=cmap, annot=test_results[['Normality', 'Homoscedasticity']],
                     fmt='', cbar=True, linewidths=1, linecolor='white', square=True, annot_kws={"size": 11, "weight": "bold"},
                     cbar_kws={"ticks": [0, 1], "label": "Test Result (0: Fail, 1: Pass)"})

        # Manually annotate the "Test" column on the left (this is the first column)
        for i, metric in enumerate(metrics):
            ax.text(-0.5, i + 0.5, tests[i], ha='center', va='center', color='black', weight='bold', fontsize=11)

        # Set the title and axis labels
        plt.title('Test Results: Normality and Homoscedasticity', fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Categories', fontsize=11, labelpad=10)
        ax.set_ylabel('Metrics', fontsize=11, labelpad=10)

        # Improve ticks visibility
        ax.set_xticklabels(['Normality', 'Homoscedasticity'], fontsize=11)

        # Format y-axis labels by removing underscores and applying .title()
        formatted_metrics = [metric.replace('_', ' ').title() for metric in metrics]
        ax.set_yticklabels(formatted_metrics, fontsize=11)

        # Adjust the layout and save the figure
        plt.tight_layout()
        
        filename = output_dir / f'{filename_pattern}_normality_homoscedasticity_heatmap.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        print(f"Heatmap saved at {filename}")


    def cohen_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d for effect size between two groups.
        """
        diff = np.mean(group1) - np.mean(group2)
        pooled_var = (np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2
        d = diff / sqrt(pooled_var)
        return d

    def eta_squared(self, anova_table: pd.DataFrame) -> float:
        """
        Calculate eta-squared effect size from ANOVA table.
        """
        return anova_table["sum_sq"][0] / (anova_table["sum_sq"][0] + anova_table["sum_sq"][1])

    def visualize_effect_sizes(self, metrics: List[str], effect_sizes: Dict[str, Union[float, None]], output_dir: Path, filename_pattern: str):
        """
        Visualize effect sizes using a scatter plot with improved aesthetics.
        """

        # Filter out metrics with None effect sizes
        valid_metrics = [metric for metric in metrics if effect_sizes[metric] is not None]
        effect_vals = [effect_sizes[metric] for metric in valid_metrics if effect_sizes[metric] is not None]


        # Check if the list contains valid numeric values
        if effect_vals and all(isinstance(val, (int, float)) for val in effect_vals):
            # Create scatter plot
            plt.figure(figsize=(10, 6))

            colors = sns.color_palette("coolwarm", len(effect_vals))  # Use a gradient color palette

            # Create scatter plot with a continuous color map and point sizes based on effect sizes
            scatter = plt.scatter(
                valid_metrics, effect_vals,
                s=[abs(val) * 500 for val in effect_vals],  # Dynamically adjust point sizes based on effect sizes
                c=effect_vals,  # Use the effect size values for color mapping
                cmap="coolwarm",  # Use a continuous color map provided by matplotlib
                edgecolor='black',  # Outline for the points
                label='Effect Sizes'
            )
            # Title and labels
            plt.title("Effect Sizes for Metrics (Scatter Plot)", fontsize=16, fontweight='bold')
            plt.ylabel("Effect Size", fontsize=14)
            plt.xlabel("Metrics", fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)

            # Add a color bar to represent the color gradient
            cbar = plt.colorbar(scatter)
            cbar.set_label('Effect Size Magnitude', fontsize=12)

            # Add gridlines
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Optionally, annotate each point with its exact effect size
            for i, (metric, val) in enumerate(zip(valid_metrics, effect_vals)):
                plt.annotate(f'{val:.4g}', (metric, val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)

            # Set axis limits (optional, can be removed if auto-scaling is preferred)
            plt.xlim(-0.5, len(valid_metrics) - 0.5)  # Keep some margin around x-axis points
            plt.ylim(min(effect_vals) - 0.05, max(effect_vals) + 0.05)  # Some margin for y-axis

            # Display the legend outside the plot
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Effect Sizes', fontsize=12)

            # Save and show plot
            plt.tight_layout()
            plt.savefig(output_dir / f'{filename_pattern}_effect_sizes_scatterplot.png')
            plt.close()

    def qq_plot(self, df: pd.DataFrame, metric: str, category: str, output_dir: Path, filename_pattern: str):
        """
        Q-Q plot for assessing normality with a legend displayed outside the plot.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            metric (str): The column name of the metric for which to generate Q-Q plots.
            category (str): The column name of the categorical variable used to group the data.
            output_dir (Path): Directory to save the plot.
            filename_pattern (str): Filename pattern for saving the output image.
        """
        
        categories = df[category].unique()
        num_categories = len(categories)
        
        plt.figure(figsize=(14, num_categories * 3))

        colors = sns.color_palette("RdBu", num_categories)
        
        # Loop through categories and create subplots
        for i, cat in enumerate(categories):
            plt.subplot(len(categories), 1, i + 1)
            sm.qqplot(df[df[category] == cat][metric], line='s', dist=norm, fit=True, ax=plt.gca(), color=colors[i])
            
            plt.title(f"Q-Q Plot for {metric} in {category}: {cat}", fontsize=12, fontweight='bold')
            plt.xlabel('Theoretical Quantiles', fontsize=11)
            plt.ylabel('Sample Quantiles', fontsize=11)
            plt.grid(True)

        # Create legend for the whole figure
        handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label=f'{cat}') for i, cat in enumerate(categories)]
        plt.gcf().legend(handles=handles, title="Categories", loc='center right', bbox_to_anchor=(1.05, 0.5), fontsize=11, title_fontsize=11)

        plt.tight_layout(rect=(0, 0, 0.92, 1.0))
        
        plt.savefig(output_dir / f'{filename_pattern}_{metric}_qqplot.png', bbox_inches='tight')
        plt.close()

        print(f"Q-Q plot saved to {output_dir / f'{filename_pattern}_{metric}_qqplot.png'}")


    def analyze_category(self, df: pd.DataFrame, category: str, filename_pattern: str, output_dir: str, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze and identify the best categories based on combined SNR, HR, and composite scores.

        This function evaluates categories within a DataFrame based on Signal-to-Noise Ratio (SNR) scores, Heart Rate (HR) scores,
        and a composite score derived from these metrics. It identifies the top N categories based on each score, determines
        the overlap between the top categories, and visualizes the results using dual-axis boxplots. The function also saves
        the analysis results and visualizations to specified output directories.

        **Output**:
            - CSV files containing the combined scores, top categories, and filtered DataFrames based on the top categories.
            - Boxplot visualizations comparing the top categories based on SNR, HR, and combined scores.

        Args:
            df (pd.DataFrame): The DataFrame containing the SNR, HR, and related data to be analyzed.
            category (str): The category column by which to group and analyze the data.
            output_dir (str): The directory where the analysis results and visualizations will be saved.
            top_n (int, optional): The number of top categories to identify based on each score. Defaults to 10.

        Returns:
            pd.DataFrame: The DataFrame containing the top N categories based on the combined score, including the necessary columns for statistical significance testing.
        """

        # Calculate the combined scores
        # scores = self.calculate_combined_scores(df, category)
        scores = self.calculate_snr_scores(df, category)

        # Sort combined scores by 'score' in descending order
        scores = scores.sort_values(by='combined_score', ascending=False)

        # Save the sorted scores to a CSV file
        scores_filename = f'{filename_pattern}_scores.csv'
        scores.to_csv(os.path.join(output_dir, scores_filename), index=False)
        
        # Print the sorted Combined Scores DataFrame
        print("Sorted Combined Scores DataFrame:")
        print(scores.head(top_n))

        df = df.merge(scores[[category, 'snr_score']], on=category, how='left')
        # df = df.merge(combined_scores[[category, 'snr_score', 'hr_score', 'combined_score', 'in_range_proportion']], on=category, how='left')

        # Sort the DataFrame by each score in descending order and get the top N
        top_snr_score = scores.nlargest(top_n, 'snr_score')
        # top_hr_score = combined_scores.nlargest(top_n, 'hr_score')
        # top_combined_score = combined_scores.nlargest(top_n, 'combined_score')

        # Extract the top N categories for each score
        top_snr_categories = top_snr_score[category].unique()
        # top_hr_categories = top_hr_score[category].unique()
        # top_combined_categories = top_combined_score[category].unique()
        # print(f"\nTop {top_n} Categories based on Combined Score: {top_combined_categories}")
        
        # Filter the DataFrame to only include the rows with top combined categories
        df_top_snr_scores = df[df[category].isin(top_snr_categories)]
        df_top_snr_scores_filename = f'{filename_pattern}_top{top_n}_snr_scores_df.csv'
        df_top_snr_scores.to_csv(os.path.join(output_dir, df_top_snr_scores_filename), index=False)
        print(f"\nFiltered DataFrame for Top {top_n} SNR Categories:")
        print(df_top_snr_scores.head(top_n))
        # Filter the DataFrame to only include the rows with top combined categories
        # df_top_combined = df[df[category].isin(top_combined_categories)]
        # df_top_combined_filename = f'{filename_pattern}_top{top_n}_combined_df.csv'
        # df_top_combined.to_csv(os.path.join(output_dir, df_top_combined_filename), index=False)
        # print(f"\nFiltered DataFrame for Top {top_n} Combined Categories:")
        # print(df_top_combined.head(top_n))

        # Print the best combination for each score
        # print(f"\nBest Based on Combined Score: {category.capitalize()} {top_combined_score.iloc[0][category]}")
        print(f"Best Based on SNR Score: {category.capitalize()} {top_snr_score.iloc[0][category]}")
        # print(f"Best Based on HR Score: {category.capitalize()} {top_hr_score.iloc[0][category]}\n")

        # Save the top scores to a CSV file
        # combined_scores_filename = f'{filename_pattern}_top{top_n}_combined_categories.csv'
        # top_combined_score.to_csv(os.path.join(output_dir, combined_scores_filename), index=False)

        snr_scores_filename = f'{filename_pattern}_top{top_n}_snr_categories.csv'
        top_snr_score.to_csv(os.path.join(output_dir, snr_scores_filename), index=False)

        # hr_scores_filename = f'{filename_pattern}_top{top_n}_hr_categories.csv'
        # top_hr_score.to_csv(os.path.join(output_dir, hr_scores_filename), index=False)
        
        # del combined_scores, top_combined_score, top_snr_score, top_hr_score
        # gc.collect()
        
        # # Take the intersection of top N categories across different scores
        # combined_snr_common = set(top_combined_categories).intersection(top_snr_categories)
        # combined_hr_common = set(top_combined_categories).intersection(top_hr_categories)
        # snr_hr_common = set(top_snr_categories).intersection(top_hr_categories)
        # combined_hr_snr_common = combined_snr_common.intersection(top_hr_categories)

        # # Print the common categories for each combination and Convert sets to DataFrame and save to CSV
        # def save_common_categories(common_set, description, filename):
        #     if common_set:
        #         categories = ', '.join(str(cat) for cat in common_set)
        #         print(f"Common categories {description}: {categories}")
        #         df_common = pd.DataFrame({description: list(common_set)})
        #         df_common.to_csv(filename, index=False)
        #     else:
        #         print(f"No common categories {description}")

        # # Create file paths
        # combined_snr_common_filename = os.path.join(output_dir, f"{filename_pattern}_combined_snr_common.csv")
        # combined_hr_common_filename = os.path.join(output_dir, f"{filename_pattern}_combined_hr_common.csv")
        # snr_hr_common_filename = os.path.join(output_dir, f"{filename_pattern}_snr_hr_common.csv")
        # combined_hr_snr_common_filename = os.path.join(output_dir, f"{filename_pattern}_combined_hr_snr_common.csv")

        # save_common_categories(combined_snr_common, "Combined_SNR_Common", combined_snr_common_filename)
        # save_common_categories(combined_hr_common, "Combined_HR_Common", combined_hr_common_filename)
        # save_common_categories(snr_hr_common, "SNR_HR_Common", snr_hr_common_filename)
        # save_common_categories(combined_hr_snr_common, "Combined_HR_SNR_Common", combined_hr_snr_common_filename)
        # print()

        # # Create a visualizations directory
        # vis_output_dir = Path(output_dir) / 'visualizations'
        # vis_output_dir.mkdir(parents=True, exist_ok=True)

        # Filter the DataFrame for top combined categories once
        df_filtered = df[df[category].isin(top_snr_categories)].copy()
        # df_filtered = df[df[category].isin(top_combined_categories)].copy()

        # # Perform correlation analysis
        # corr_columns = ['snr', 'heart_rate', 'in_range_proportion', 'combined_score']
        # self.calculate_and_visualize_correlations(df_filtered[corr_columns].drop_duplicates(), vis_output_dir, filename_pattern)

        # # Visualize correlation between SNR and HR
        # snr_hr_columns = ['snr', 'heart_rate']
        # self.calculate_and_visualize_snr_hr_correlation(df_filtered[snr_hr_columns].drop_duplicates(), vis_output_dir, filename_pattern)

        # # Outlier detection and metric correlation with category
        # outlier_columns = [category, 'snr', 'heart_rate']
        # self.detect_outliers(df_filtered[outlier_columns].drop_duplicates(), ['snr', 'heart_rate'], category, vis_output_dir, filename_pattern)
        # self.calculate_metrics_correlation_with_category(df_filtered[outlier_columns], category, vis_output_dir, filename_pattern)

        # # Remove columns no longer needed to free memory
        # df_filtered.drop(columns=outlier_columns[1:], inplace=True)  # Drop 'snr' and 'heart_rate' to save memory
        # gc.collect()

        # # Visualize variability and impact using only relevant columns
        # score_columns = [category, 'snr_score', 'hr_score', 'in_range_proportion', 'combined_score']
        # df_scores = df_filtered[score_columns].drop_duplicates()
        # self.visualize_variability(df_scores, 'snr_score', 'hr_score', 'in_range_proportion', 'combined_score', category, vis_output_dir, filename_pattern)
        # self.visualize_variability_impact(df_scores, 'snr_score', 'hr_score', 'in_range_proportion', 'combined_score', category, vis_output_dir, filename_pattern)
        # self.bar_plot_combined_scores(df_scores, category, 'combined_score', vis_output_dir, filename_pattern, top_n)
        # self.heatmap_correlation(df_scores, 'snr_score', 'hr_score', 'in_range_proportion', 'combined_score', vis_output_dir, filename_pattern)
        # self.bar_plot_scores(df_scores, category, 'snr_score', 'hr_score', 'in_range_proportion', 'combined_score', vis_output_dir, filename_pattern)

        # for metric in ['snr', 'heart_rate']:
        #     df_filtered[metric] = df[metric]

        #     # Plot the distribution of both SNR and HR across different categories in a grid format
        #     self.visualize_facet_grid(df_filtered, metric, category, vis_output_dir, filename_pattern)
            
        #     # Remove the 'metric' column after visualizing to free memory
        #     df_filtered.drop(columns=[metric], inplace=True)
        #     gc.collect()

        # print('\n')

        return df_top_snr_scores

    def check_statistical_significance(self, df: pd.DataFrame, category: Union[str, list], top_combinations: pd.DataFrame, output_dir: str, filename_pattern: str,  alpha: float = 0.05) -> dict:
        """
        Check the statistical significance of signal processing metrics (SNR and Heart Rate) across different categories.

        This function evaluates whether there are statistically significant differences in SNR and Heart Rate across the top combinations
        of the specified category or categories. It first checks for normality and homogeneity of variance to determine the appropriate statistical test
        (ANOVA or Kruskal-Wallis). The function returns the p-values and test results for each metric.

        **Process**:
            1. Ensure the `category` argument is a list.
            2. Identify the unique groups from the `top_combinations` DataFrame.
            3. Merge the main DataFrame with the unique groups to focus on top combinations.
            4. Group the data by the specified categories and prepare lists for statistical tests.
            5. Check the normality assumption using the Shapiro-Wilk or Kolmogorov-Smirnov test.
            6. Check the homogeneity of variance assumption using Levene's test.
            7. Depending on the assumptions, perform either ANOVA or the Kruskal-Wallis test.
            8. Determine if the differences are statistically significant based on the p-value.
            9. Print the results and return them in a dictionary.

        Args:
            df (pd.DataFrame): The DataFrame containing the signal processing metrics and parameters to analyze.
            category (str or list): The category or categories to group by for statistical analysis.
            top_combinations (pd.DataFrame): A DataFrame containing the top combinations to analyze for statistical significance.
            output_dir (Union[str, Path]): The directory where visualizations will be saved.
            filename_pattern (str): Pattern to use for saving the visualizations.
            alpha (float, optional): The significance level for hypothesis testing. Defaults to 0.05.

        Returns:
            dict: A dictionary where keys are metric names ('snr', 'heart_rate') and values are dictionaries containing:
                - 'p_value': The p-value from the statistical test.
                - 'test_used': The name of the statistical test used (ANOVA or Kruskal-Wallis).
                - 'normality': A boolean indicating if the data met the normality assumption.
                - 'homoscedasticity': A boolean indicating if the data met the homoscedasticity assumption.
        
        Example:
            results = check_statistical_significance(df, category='region', top_combinations=top_combinations)
        """

        results: Dict[str, Dict[str, Union[float, str, bool]]] = {}
        effect_sizes: Dict[str, Union[float, None]] = {}
        metrics: List[str] = ['snr', 'heart_rate']

        # Ensure category is a list
        if isinstance(category, str):
            category = [category]
        
        # Get unique groups from top_combinations
        unique_groups = top_combinations[category].drop_duplicates()

        # Filter the main dataframe for only the top combinations
        merged_df = pd.merge(df, unique_groups, on=category, how='inner')

        for metric in metrics:
            result: Dict[str, Union[float, str, bool]] = {}
            # Group data by the specified categories
            grouped_data = merged_df.groupby(category)[metric].apply(list)

            # Prepare data for tests
            data_lists = [values for values in grouped_data]

            # Check if there are at least two groups to compare
            if len(data_lists) < 2:
                print(f"Not enough groups for statistical comparison of {metric} in category {category}. Skipping test.")
                result['p_value'] = None
                result['test_used'] = 'Not enough groups'
                result['normality'] = None
                result['homoscedasticity'] = None
                results[metric] = result
                continue

            # Check assumptions for ANOVA
            # 1. Normality: Use Shapiro-Wilk for N <= 5000, else use Kolmogorov-Smirnov
            normality_pvals = []
            for values in data_lists:
                if len(values) > 5000:
                    normality_pvals.append(stats.kstest(values, 'norm')[1])
                else:
                    normality_pvals.append(shapiro(values)[1])
            normal = all(p > alpha for p in normality_pvals)

            # 2. Homogeneity of Variance (Levene's Test)
            if len(data_lists) >= 2 and all(len(values) >= 2 for values in data_lists):
                levene_stat, levene_p = stats.levene(*data_lists)
                homoscedastic = levene_p > alpha
            else:
                homoscedastic = False  # Not enough data to test

            # Decide which test to use
            if normal and homoscedastic and len(data_lists) >= 2:
                # Perform One-Way ANOVA
                try:
                    formula = f'{metric} ~ ' + ' + '.join([f'C({cat})' for cat in category])
                    model = ols(formula, data=merged_df).fit()
                    anova_results = anova_lm(model, typ=2)
                    p_value = anova_results["PR(>F)"][0]
                    test_used = 'ANOVA'
                    effect_sizes[metric] = self.eta_squared(anova_results)  # Calculate eta-squared effect size

                except Exception as e:
                    print(f"ANOVA failed for {metric} due to {e}. Falling back to Kruskal-Wallis test.")
                    stat, p_value = kruskal(*data_lists)
                    test_used = 'Kruskal-Wallis'
                    effect_sizes[metric] = None  # No effect size for Kruskal-Wallis
            else:
                # Perform Kruskal-Wallis Test
                stat, p_value = kruskal(*data_lists)
                test_used = 'Kruskal-Wallis'
                if len(data_lists) == 2:
                    effect_sizes[metric] = self.cohen_d(data_lists[0], data_lists[1])  # Calculate Cohen's d for two groups
                else:
                    effect_sizes[metric] = None  # No effect size for Kruskal-Wallis with multiple groups

                # Run Dunn's post-hoc test
                post_hoc_results = sp.posthoc_dunn(merged_df, val_col=metric, group_col=category[0], p_adjust='bonferroni')
                result['post_hoc_results'] = post_hoc_results

            # Determine statistical significance
            is_significant = p_value < alpha

            # Store results
            result['p_value'] = p_value
            result['test_used'] = test_used
            result['normality'] = normal
            result['homoscedasticity'] = homoscedastic
            results[metric] = result

            # Print the results
            significance_statement = "Statistically Significant" if is_significant else "Not Statistically Significant"
            category_str = ', '.join(category)
            print(f"Category: {category_str}, Metric: {metric.capitalize()} - {significance_statement} (p-value: {p_value:.4g}) using {test_used}")
        
        print()

        vis_output_dir: Path = Path(output_dir) / 'visualizations'
        vis_output_dir.mkdir(parents=True, exist_ok=True)

        # Visualize statistical significance
        self.visualize_test_results(results, metrics, vis_output_dir, filename_pattern)

        # Visualize effect sizes
        self.visualize_effect_sizes(metrics, effect_sizes, vis_output_dir, filename_pattern)
        
        # Visualize Q-Q plots for normality, and Dunn's post-hoc test
        for metric in metrics:
            self.qq_plot(merged_df, metric, category[0], vis_output_dir, filename_pattern)

            self.visualize_post_hoc_results(post_hoc_results=results[metric]['post_hoc_results'],
                                    metric=metric,
                                    category=category[0],
                                    output_dir=vis_output_dir,
                                    filename_pattern=filename_pattern)

        return results

    def analyze_best_combinations(self, df: pd.DataFrame, filename_pattern: str, top_n: int = 3):
        """
        Analyze and identify the best combinations of signal processing parameters across various categories.
        
        This function evaluates different combinations of categories such as region, ROI type, signal type,
        wavelet, filter, and others. For each category, it identifies the top N combinations based on specific
        criteria, checks their statistical significance, and saves the results for further analysis.

        **Process**:
            1. Defines a set of categories and their corresponding columns for analysis.
            2. For each category, it combines columns if necessary, then identifies and ranks the top N combinations.
            3. Checks the statistical significance of these top combinations.
            4. Saves the results and significance data to CSV files.
            5. Visualizes the statistical significance of the identified top combinations.

        **Output**:
            - CSV files for the best combinations in each category.
            - CSV files for the statistical significance of the top combinations.
            - Visualizations of statistical significance saved to the output directory.

        Args:
            df (pd.DataFrame): The DataFrame containing signal processing metrics and parameters to analyze.
            top_n (int, optional): The number of top combinations to identify and analyze for each category. Defaults to 3.

        Returns:
            None: The function saves the analysis results and visualizations to the specified output directory.
        """
        
        base_output_dir = os.path.join(path_config.SNR_ANALYZER_OUTPUT_DIR.parent.parent.parent, "analyze_best_combinations")
        os.makedirs(base_output_dir, exist_ok=True)

        categories_to_analyze: Dict[str, List[str]] = {
            # 'region': ['region'],
            # 'roi': ['roi_type'],
            # 'video': ['video_name'],

            # 'roi_region': ['roi_type', 'region'],
            # 'roi_region_video': ['roi_type', 'region', 'video_name'],
            # 'roi_video': ['roi_type', 'video_name'],
            # 'region_video': ['region', 'video_name'],

            # 'region_patch': ['region', 'patch_idx'],
            # 'region_patch_video': ['region', 'patch_idx', 'video_name'],
            # 'roi_region_patch': ['roi_type', 'region', 'patch_idx'],
            # 'signal': ['signal_type'],
            # 'signal_video': ['signal_type', 'video_name'],
            # 'wavelet': ['wavelet'],
            # 'wavelet_level': ['wavelet', 'wavelet_level'],
            # 'filter': ['filter'],
            # 'filter_order': ['filter', 'filter_order'],
            # 'wavelet_level_filter_order': ['wavelet', 'wavelet_level', 'filter', 'filter_order'],
            # 'signal_wavelet_level': ['signal_type', 'wavelet', 'wavelet_level'],
            'signal_filter_order': ['signal_type', 'filter', 'filter_order'],
            'signal_wavelet_level_filter_order': ['signal_type', 'wavelet', 'wavelet_level', 'filter', 'filter_order'],
        }

        for category_name, grouping_columns in categories_to_analyze.items():
            
            print(f"\nAnalyzing category: {category_name} with columns {grouping_columns}")

            # If grouping_columns has multiple elements, combine them into a single column for analysis
            if len(grouping_columns) > 1:
                combined_column_name = '_'.join(grouping_columns)
                df[combined_column_name] = df[grouping_columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
                category_to_analyze = combined_column_name
            else:
                category_to_analyze = grouping_columns[0]

            category_output_dir = os.path.join(base_output_dir, category_to_analyze)
            os.makedirs(category_output_dir, exist_ok=True)
            print(f"Output directory for category '{category_name}': {category_output_dir}")

            # Analyze category and Get the top N combinations
            # top_combinations = self.analyze_category(df, category_to_analyze, filename_pattern, category_output_dir, top_n=top_n)
            self.analyze_category(df, category_to_analyze, filename_pattern, category_output_dir, top_n=top_n)
            
            # if not top_combinations.empty:
            #     print(f"Top {top_n} combinations for category '{category_name}' found.")
                
            #     significance = self.check_statistical_significance(df, category_to_analyze, top_combinations, category_output_dir, filename_pattern)
            #     print(f"Significance for category '{category_name}':\n", significance)

            #     # Save results to CSV
            #     top_combinations_filename = os.path.join(category_output_dir, f'{filename_pattern}_best_{category_name}_combinations.csv')
            #     top_combinations.to_csv(top_combinations_filename, index=False)
            #     print(f"Saved top combinations to: {top_combinations_filename}")

            #     # Save significance results
            #     significance_df = pd.DataFrame(significance).T.reset_index().rename(columns={'index': 'metric'})
            #     significance_filename = os.path.join(category_output_dir, f'{filename_pattern}_significance_{category_name}.csv')
            #     significance_df.to_csv(significance_filename, index=False)
            #     print(f"Saved significance results to: {significance_filename}")

            # else:
            #     print(f"No valid combinations found for category {category_name}")


def process_file(file_path: Path) -> pd.DataFrame:
    """
    Process a single CSV file by reading its contents into a DataFrame.

    Args:
        file_path (Path): The path to the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    try:
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=50000):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def get_files_by_parent_folder(base_dir: Union[str, Path], filename_pattern: str) -> Dict[Path, List[Path]]:
    """
    Get a dictionary where each key is a parent folder and the value is a list of files 
    in that folder matching the filename pattern.

    Args:
        base_dir (Union[str, Path]): The base directory to search for result files.
        filename_pattern (str): The pattern to match filenames (e.g., 'snr_optimization_*.h5').

    Returns:
        Dict[Path, List[Path]]: A dictionary with parent folders as keys and lists of matching files as values.
    """
    base_dir = Path(base_dir)
    filename_pattern = filename_pattern + '*.h5'
    
    # Get all result files matching the pattern
    result_files = list(base_dir.rglob(filename_pattern))
    
    # Dictionary to hold files grouped by their parent folder
    files_by_parent_folder = {}

    for file in result_files:
        parent_folder = file.parent
        
        # Initialize the list for this parent folder if it doesn't exist
        if parent_folder not in files_by_parent_folder:
            files_by_parent_folder[parent_folder] = []
        
        # Append the current file to the list of its parent folder
        files_by_parent_folder[parent_folder].append(file)

    return files_by_parent_folder

def _load_group(group: h5py.Group) -> Dict[str, Any]:
    """
    Recursively load the contents of an HDF5 group into a dictionary.

    Args:
        group (h5py.Group): The HDF5 group to load.

    Returns:
        Dict[str, Any]: A dictionary containing the group's data.
    """
    group_dict: Dict[str, Any] = {}

    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            # Directly load the dataset into the dictionary
            group_dict[key] = item[()]
        elif isinstance(item, h5py.Group):
            # Recursively load the subgroup and store it in the dictionary
            group_dict[key] = _load_group(item)

    # Load group attributes, if any
    group_dict.update({key: group.attrs[key] for key in group.attrs})

    return group_dict

def _process_file(result_file: Path) -> List[Dict[str, Any]]:
    results = []
    with h5py.File(result_file, 'r') as f:
        for key in f.keys():
            group = f[key]
            result_dict = _load_group(group)

            # Add metadata
            file_path = Path(f.filename)
            result_dict['video_name'] = file_path.parent.name
            result_dict['roi_type'] = file_path.parent.parent.name
            result_dict['subfolder'] = f"{file_path.parent.parent.parent.parent.parent.name}_{file_path.parent.parent.parent.parent.name}"

            results.append(result_dict)
    return results



if __name__ == "__main__":
    snr_analyzer = SNRAnalyzer()

    subfolders: List[str] = ['real/train', 'real/val', 'real/test']
    roi_types: List[str] = ['mask', 'patch']

    path_config.DATASET = 'GoogleDFD'
    path_config.METHOD = 'rgb'
    path_config.update_paths()

    # snr_analyzer.process_and_save_results()
    filename_pattern = '_1_' # _0_, _1_, _2_
    df = snr_analyzer.load_results(filename_pattern)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    df = df[(df['snr'] != 0) & (df['heart_rate'] != 0)] # (df['heart_rate'] >= 60) & (df['heart_rate'] <= 100) & (df['snr'] != 1000) &

    top_n = 20
    snr_analyzer.analyze_best_combinations(df, filename_pattern, top_n=top_n)

    # 1. Categorical Analysis (Analyze SNR/HR by Signal Type/...)
    # 2. Correlation Analysis
    # 3. Comparative Analysis
    # 4. Visualization