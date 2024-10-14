"""
This module provides functionality for handling and processing video files. The primary class is VideoHandler, which uses VideoProcessor to process videos and save the results.

Classes:
    VideoHandler: Handles the processing and saving of video processing files.
"""

import os
import gc
import logging
from pathlib import Path
import json
from typing import Dict, List, Any
import cv2
import numpy as np
from rppg_facepatches.video_processor import path_config
from rppg_facepatches.video_processor.video_processor import VideoProcessor

logging.getLogger().setLevel(logging.WARNING)

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class VideoHandler:
    """
    A class used to process and save video files.

    Attributes:
        video_processor (VideoProcessor): An instance of the VideoProcessor class used to process videos.

    Methods:
        process_and_save_video(face_detector):
            Processes a single video file and saves the results.
        process_and_save_videos(video_list: List[str], project_path: Path, subfolder: str, method: str, process_type: str, face_detector):
            Processes and saves a list of video files.
        _validate_processed_data(normalized_values_by_region: Dict[str, Dict[str, np.ndarray]], pulse_signals_by_region: Dict[str, Dict[str, np.ndarray]], snr_values_by_region: Dict[str, Dict[str, np.ndarray]], hr_values_by_region: Dict[str, Dict[str, np.ndarray]], framerate: float | None) -> bool:
            Validates the processed video data to ensure it has the expected format and contents.
        _convert_video_results_to_list(normalized_values_by_region: Dict[str, Dict[str, np.ndarray]], pulse_signals_by_region: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, Dict[str, list]], Dict[str, Dict[str, list]]]:
            Converts numpy arrays in the video results to lists for JSON serialization.
        _save_video_results(normalized_values_by_region: Dict[str, Dict[str, np.ndarray]], pulse_signals_by_region: Dict[str, Dict[str, np.ndarray]], snr_values_by_region: Dict[str, Dict[str, np.ndarray]], hr_values_by_region: Dict[str, Dict[str, np.ndarray]], framerate: float | None):
            Saves the video processing results into separate JSON files.
    """

    def __init__(self) -> None:
        """
        Initializes the Processor with a VideoProcessor instance.
        """
        self.video_processor = VideoProcessor()

    def process_and_save_video(self, face_detector, video_path, index, subfolder, method, roi_type):
        """
        Processes a single video file and saves the results.

        This method configures paths and settings for video processing, 
        checks if the output directory already contains results, processes the video 
        to extract various features, validates the processed data, and saves the results if valid.

        Args:
            video_path (Path): The path to the video file to be processed.
            index (int): The index of the video, used for naming and organizing output.
            subfolder (str): The subfolder name where results will be stored.
            method (str): The method used for processing the video.
            roi_type (str): The region of interest type for processing the video.

        Returns:
            None: This method does not return any value. It either saves the processed 
            video data to the configured output directory or exits if the data is invalid 
            or already processed.
        """
        path_config.SUBFOLDER = subfolder
        path_config.METHOD = method
        path_config.ROI_TYPE = roi_type
        path_config.VIDEO_FILE_PATH = video_path
        path_config.VIDEO_INDEX = index
        path_config.DATASET = video_path.parts[2]
        path_config.PULSE_SIGNALS_FILE_NAME = f'pulse_signals_{path_config.VIDEO_INDEX}_{path_config.VIDEO_FILE_PATH.stem}.npy'
        path_config.update_paths()
        
        # self.display_video_frames()
        # self.save_video_frames()

        # Check if the output directory for this video already exists and is not empty
        if os.path.exists(path_config.VIDEO_OUTPUT_DIR):
            # Check if the PULSE_SIGNALS_FILE_NAME already exists in the output directory
            if path_config.PULSE_SIGNALS_FILE_NAME in os.listdir(path_config.VIDEO_OUTPUT_DIR):
                gc.collect()
                return
            
        # Process the video to extract features
        normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate = self.video_processor.process_video(face_detector)

        # if not self._validate_processed_data(normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate):
        #     del normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate
        #     gc.collect()
        #     return

        # # Save the results
        # self._save_video_results(normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate)

        # gc.collect()

    def _validate_processed_data(
        self,
        normalized_values_by_region: Dict[str, Dict[str, np.ndarray]],
        pulse_signals_by_region: Dict[str, Dict[str, np.ndarray]],
        snr_values_by_region: Dict[str, Dict[str, np.ndarray]],
        hr_values_by_region: Dict[str, Dict[str, np.ndarray]],
        framerate: float | None,
    ) -> bool:
        """
        Validates the processed video data to ensure it has the expected format and contents.

        Args:
            normalized_values_by_region (Dict[str, Dict[str, float]]): Normalized color values by region.
            pulse_signals_by_region (Dict[str, Dict[str, float]]): Pulse signals by region.
            snr_values_by_region (Dict[str, Dict[str, float]]): Signal-to-noise ratio values by region.
            hr_values_by_region (Dict[str, Dict[str, float]]): Heart rate values by region.
            framerate (float): The framerate of the processed video.

        Returns:
            bool: True if all data is valid, False otherwise.
        """
        if not all(
            isinstance(data, dict)
            for data in [
                normalized_values_by_region,
                pulse_signals_by_region,
                snr_values_by_region,
                hr_values_by_region,
            ]
        ):
            return False

        if not isinstance(framerate, (int, float)) or framerate <= 0:
            return False

        if any(
            not data
            for data in [
                normalized_values_by_region,
                pulse_signals_by_region,
                snr_values_by_region,
                hr_values_by_region,
            ]
        ):
            return False

        for region in [
            normalized_values_by_region,
            pulse_signals_by_region,
            snr_values_by_region,
            hr_values_by_region,
        ]:
            if not all(
                isinstance(key, str) and bool(value) for key, value in region.items()
            ):
                return False

        return True

    def _convert_arrays_to_list(self, data: Any):
        """
        Convert all numpy arrays in the given data structure to lists.

        Args:
            data (any): A data structure containing numpy arrays.

        Returns:
            any: The data structure with all numpy arrays converted to lists.
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._convert_arrays_to_list(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_arrays_to_list(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._convert_arrays_to_list(item) for item in data)
        else:
            return data

    def _save_video_results(
        self,
        normalized_values: Dict[str, Dict[str, np.ndarray]],
        pulse_signals: Dict[str, Dict[str, np.ndarray]],
        snr_values: Dict[str, Dict[str, np.ndarray]],
        hr_values: Dict[str, Dict[str, np.ndarray]],
        framerate: float | None,
    ):
        """
        Saves the video processing results into separate JSON files.

        Args:
            output_path (str): The directory where results will be saved.
            video_file (str): The name of the processed video file.
            normalized_values_by_region (Dict[str, Dict[str, float]]): Normalized color values by region.
            pulse_signals_by_region (Dict[str, Dict[str, float]]): Pulse signals by region.
            snr_values_by_region (Dict[str, Dict[str, float]]): Signal-to-noise ratio values by region.
            hr_values_by_region (Dict[str, Dict[str, float]]): Heart rate values by region.
            framerate (float): The framerate of the processed video.
        """
        # Convert numpy arrays to lists before saving
        # normalized_values = self._convert_arrays_to_list(normalized_values)
        pulse_signals = self._convert_arrays_to_list(pulse_signals)
            
        # Ensure the output directory exists
        path_config.VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # # Save the framerate
        # framerate_path = path_config.VIDEO_OUTPUT_DIR / "framerate.json"
        # with open(framerate_path, "w", encoding="utf-8") as f:
        #     json.dump({"framerate": framerate}, f, indent=4)

        # # Save the normalized values by region
        # normalized_values_path = path_config.VIDEO_OUTPUT_DIR / "normalized_values.json"
        # with open(normalized_values_path, "w", encoding="utf-8") as f:
        #     json.dump(normalized_values, f, indent=4)

        # Save the pulse signals by region
        pulse_signals_path = path_config.VIDEO_OUTPUT_DIR / path_config.PULSE_SIGNALS_FILE_NAME
        np.save(pulse_signals_path, pulse_signals)
        # with open(pulse_signals_path, "w", encoding="utf-8") as f:
        #     json.dump(pulse_signals, f, indent=4)

        # # Save the SNR values by region
        # snr_values_path = path_config.VIDEO_OUTPUT_DIR / "snr_values.json"
        # with open(snr_values_path, "w", encoding="utf-8") as f:
        #     json.dump(snr_values, f, indent=4)

        # # Save the HR values by region
        # hr_values_path = path_config.VIDEO_OUTPUT_DIR / "hr_values.json"
        # with open(hr_values_path, "w", encoding="utf-8") as f:
        #     json.dump(hr_values, f, indent=4)

    def display_video_frames(self):
        """
        Displays all frames from the video file in a playback window.

        The frames are displayed one by one in a window until the end of the video or until 'q' is pressed.
        """
        # Open the video file for playback
        cap = cv2.VideoCapture(str(path_config.VIDEO_FILE_PATH))
        if not cap.isOpened():
            print(f"Error: Could not open video file {path_config.VIDEO_FILE_PATH}")
            return

        frame_index = 0  # Initialize frame index

        # Display each frame until the end of the video or until 'q' is pressed
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video file reached after {frame_index} frames.")
                break

            # Display the frame in a window
            cv2.imshow('Video Playback', frame)

            # Print current frame index for debugging
            print(f"Displaying Frame: {frame_index}")

            frame_index += 1  # Increment the frame index

            # Wait for 25 ms before displaying the next frame
            # If 'q' is pressed, break the loop and stop playback
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("Playback stopped by user.")
                break

        # Release video capture and close display window
        cap.release()
        cv2.destroyAllWindows()

    def save_video_frames(self):
        """
        Saves all frames from the video file without displaying it.

        The frames are saved under the following path format:
        frame_{frame_index}_{VIDEO_INDEX}_{VIDEO_FILENAME}.jpg
        """
        # Open the video file for playback
        cap = cv2.VideoCapture(str(path_config.VIDEO_FILE_PATH))
        if not cap.isOpened():
            print(f"Error: Could not open video file {path_config.VIDEO_FILE_PATH}")
            return

        # Create the directory to save frames if it doesn't exist
        frames_output_dir = path_config.VIDEO_OUTPUT_DIR / 'frames'
        os.makedirs(frames_output_dir, exist_ok=True)

        frame_index = 0  # Initialize frame index

        # Process and save frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Construct the filename and path for each frame
            filename = f'frame_{frame_index}_{path_config.VIDEO_INDEX}_{path_config.VIDEO_FILE_PATH.stem}.jpg'
            save_path = frames_output_dir / filename

            # Save the current frame as an image file
            cv2.imwrite(str(save_path), frame)
            print(f"Saved: {save_path}")  # Print to confirm saving

            frame_index += 1  # Increment the frame index

        # Release video capture and close display window
        cap.release()
        
        # Print confirmation message
        print(f"All {frame_index} frames have been saved successfully to {frames_output_dir}")


    def process_and_save_videos(
        self,
        video_list: List[Path],
        subfolder: str,
        method: str,
        roi_type: str,
        face_detector,
    ):
        """
        Processes and saves a list of video files.

        Args:
            video_list (List[Path]): List of video file paths to be processed.
            subfolder (str): The subfolder for saving the results.
            method (str): The method used for processing ('orgb' or other).
            roi_type (str): The region of interest type used for processing.
            face_detector: The face detector used in video processing.
        """
        path_config.SUBFOLDER = subfolder
        path_config.METHOD = method
        path_config.ROI_TYPE = roi_type
        
        for index, video_file in enumerate(video_list):
            path_config.VIDEO_FILE_PATH = video_file
            path_config.VIDEO_INDEX = index
            path_config.VIDEO_OUTPUT_DIR, path_config.SNR_OPTIMIZER_OUTPUT_DIR, path_config.SIGNAL_ANALYZER_OUTPUT_DIR = path_config.update_paths()

            # Check if the output directory for this video already exists and is not empty
            if os.path.exists(path_config.VIDEO_OUTPUT_DIR) and os.listdir(path_config.VIDEO_OUTPUT_DIR):
                gc.collect()
                continue
            
            # Process the video to extract features
            normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate = self.video_processor.process_video(face_detector)

            if not self._validate_processed_data(normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate):
                del normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate
                gc.collect()
                continue

            # Save the results
            self._save_video_results(normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate)

            gc.collect()