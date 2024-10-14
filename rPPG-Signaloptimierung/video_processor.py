"""
This module provides functionality for processing video files, including face tracking, mask creation, and pulse signal analysis.

Classes:
    VideoProcessor: Handles the processing of video files for face tracking, mask creation, and pulse signal analysis.
"""

import os
import gc
from typing import Optional, Tuple, Dict, List, Any, Union
from tqdm import tqdm
import numpy as np
import cv2
import mediapipe as mp
from rppg_facepatches.video_processor import mp_config
from rppg_facepatches.video_processor import path_config
from rppg_facepatches.video_processor.face_tracker import FaceTracker
from rppg_facepatches.video_processor.mask_manager import MaskManager
from rppg_facepatches.video_processor.pulse_signal_processor import PulseSignalProcessor
from rppg_facepatches.video_processor.face_detector import FaceDetector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VideoProcessor:
    """
    A class used to process video files for face tracking, mask creation, and pulse signal analysis.

    Attributes:
        face_tracker (FaceTracker): An instance of the FaceTracker class used for tracking faces.
        mask_manager (MaskManager): An instance of the MaskManager class used for creating masks.
        data_splitter (DataSplitter): An instance of the DataSplitter class used for splitting video data.
        segment_size (int): The size of each video segment.
        overlap (int): The overlap size between video segments.

    Methods:
        __init__(segment_size: int = 180, overlap: int = 25, num_patches: int = 10):
            Initializes the VideoProcessor with the specified segment size, overlap, and number of patches.
        process_video(face_detector) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], float | None]:
            Processes the video file for face tracking, mask creation, and pulse signal analysis.
        _open_video() -> Tuple[Optional[cv2.VideoCapture], Optional[float]]:
            Opens the video file at the given path and retrieves its framerate.
        _update_normalized_values_and_pulse_signals(segment_pulse_signals: Dict[str, Dict[str, np.ndarray]], segment_color_values: Dict[str, np.ndarray], segment_snr_values: Dict[str, np.ndarray], segment_hr_values: Dict[str, np.ndarray], pulse_signals: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]], normalized_color_values: Dict[str, Dict[str, List[np.ndarray]]], snr_values: Dict[str, Dict[str, List[float]]], hr_values: Dict[str, Dict[str, List[float]]]):
            Updates the normalized values and pulse signals with the calculated values for the current segment.
        _retain_overlap_data(segment_data: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
            Retains the overlap data for the next segment processing.
        _visualize_frame(rgb_frame: np.ndarray, masks: Dict[str, np.ndarray], output_folder: str, frame_idx: int, process_type: str):
            Visualizes the masks on the RGB frame and saves the visualized frames.
    """

    def __init__(self, segment_size: int = 128, overlap: int = 32, num_patches: int = 10):
        """
        Initializes the VideoProcessor with the specified segment size, overlap, and number of patches.

        Args:
            segment_size (int): The size of each video segment. Defaults to 50.
            overlap (int): The overlap size between video segments. Defaults to 25.
            num_patches (int): The number of patches for the mask manager. Defaults to 10.
        """
        self.face_tracker = FaceTracker()
        self.mask_manager = MaskManager(num_patches=num_patches)
        self.face_detector: Optional[FaceDetector] = None

        self.segment_size = segment_size
        self.overlap = overlap
    
    def open_video(self) -> Tuple[Optional[cv2.VideoCapture], Optional[float]]:
        """
        Opens the video file at the given path and retrieves its framerate.

        Args:
            video_path (str): The path to the video file.

        Returns:
            Tuple[Optional[cv2.VideoCapture], Optional[float]]: A tuple containing the video capture object and the framerate.
            Returns (None, None) if the video file cannot be opened.
        """
        cap = cv2.VideoCapture(str(path_config.VIDEO_FILE_PATH))
        if not cap.isOpened():
            return None, None
        framerate = cap.get(cv2.CAP_PROP_FPS)
        return cap, framerate

    def process_video(self, face_detector: FaceDetector, tqdm_position: int = 4) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], float | None]:
        """
        Processes the video file for face tracking, mask creation, and pulse signal analysis.

        Args:
            face_detector: The face detector used in video processing.

        Returns:
            Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], float|None]:
            A tuple containing dictionaries of normalized color values, pulse signals, signal-to-noise-ratio values, heart rate values, and the framerate.
        """
        self.face_detector = face_detector
        # Check if face_detector is a valid instance of FaceDetector
        if not isinstance(self.face_detector, FaceDetector):
            raise ValueError("Invalid face_detector: Expected an instance of FaceDetector.")

        cap, framerate = self.open_video()  # Open the video file and get the framerate
        if (not cap or framerate is None):  # Check if the video capture or framerate is None
            return {}, {}, {}, {}, 0.0

        face_rect: Tuple[int, int, int, int] = (0, 0, 1, 1)  # Initialize face rectangle
        tracked_faces: dict = {}  # Dictionary to store tracked faces
        next_face_id: int = 0  # Initialize the next face ID
        last_kp_coord: Optional[Tuple[int, int]] = None  # Initialize the last known key point coordinates
        face_landmarks = None

        # Define the keys for pulse signals based on the is_orgb flag
        pulse_signal_keys: List[str] = (["oRG", "oYB", "oRGYB", 'aT', 'o2SR'] if path_config.METHOD == "orgb"
                                        else ["R", "G", "B", "RG", "RB", "GB", "RGB", 'APOS', 'POS', '2SR'])

        # Initialize dictionaries to store results
        pulse_signals_by_region: Dict[str, Dict[str, Any]] = {}
        normalized_values_by_region: Dict[str, Dict[str, Any]] = {}
        snr_values_by_region: Dict[str, Dict[str, Any]] = {}
        hr_values_by_region: Dict[str, Dict[str, Any]] = {}

        pulse_signal_processor = PulseSignalProcessor(fps=framerate, pulse_signals_keys=pulse_signal_keys)  # Initialize the pulse signal processor
        n_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the number of frames in the video
        segment_data: Dict[str, List[np.ndarray]] = {"frames": []}  # Initialize a dictionary to store segment data

        tqdm_position = mp_config.increment_shared_counter()
        with tqdm(total=n_frames, desc=f"Processing {path_config.VIDEO_INDEX} {path_config.VIDEO_FILE_PATH.name}", position=tqdm_position, leave=False, unit='frame', dynamic_ncols=True) as video_progress:
            for frame_index in range(n_frames):  # Loop through each frame in the video
                video_progress.update(1) # Update the video progress bar
                
                ret, frame = cap.read()  # Read a frame from the video
                if not ret or frame.size == 0:  # Check if the frame was read successfully
                    last_kp_coord = self.face_tracker.reset_tracking(tracked_faces, segment_data, last_kp_coord)  # Reset tracking if the frame is not valid
                    continue

                rgb_video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB format
                detected_face = self.face_tracker.detect_face(rgb_video_frame, last_kp_coord, self.face_detector, frame_idx=frame_index)  # Detect faces in the frame

                if detected_face:  # If a face is detected
                    face_rect, key_point_coord, rgb_cropped_frame, face_landmarks = detected_face

                    if path_config.ROI_TYPE != "background":
                        if face_landmarks is None:
                            last_kp_coord = self.face_tracker.reset_tracking(tracked_faces, segment_data, last_kp_coord)
                            continue
                        rgb_video_frame = rgb_cropped_frame # If the process type is not 'background', use the cropped frame as rgb frame
                        tracked_faces, next_face_id = self.face_tracker.update_tracked_faces(tracked_faces, next_face_id, face_rect, key_point_coord)
                        last_kp_coord = key_point_coord

                    if rgb_video_frame is None or rgb_video_frame.size <= 0:  # Check if the RGB frame is valid
                        last_kp_coord = self.face_tracker.reset_tracking(tracked_faces, segment_data, last_kp_coord)
                        continue

                else:
                    if path_config.ROI_TYPE != "background":
                        last_kp_coord = self.face_tracker.reset_tracking(tracked_faces, segment_data, last_kp_coord)
                        continue

                masks, rgb_video_frame = self.mask_manager.create_masks(rgb_video_frame, face_landmarks, face_rect, frame_index)  # Create masks for the frame
                if not self.validate_masks(masks):  # If masks are not created, reset tracking
                    del rgb_video_frame, masks
                    last_kp_coord = self.face_tracker.reset_tracking(tracked_faces, segment_data, last_kp_coord)
                    continue

                for region, mask in masks.items():  # Add the masks to the segment data
                    if region not in segment_data:
                        segment_data[region] = []
                    
                    if isinstance(mask, np.ndarray):
                        segment_data[region].append(mask)
                    else:
                        segment_data[region].append(np.array(mask))

                del masks

                segment_data["frames"].append(rgb_video_frame)  # Add the frame to the segment data
                del rgb_video_frame

                if (len(segment_data["frames"]) >= self.segment_size):  # If the segment size is reached, calculate pulse signals
                    segment_pulse_signals, segment_color_values, segment_snr_values, segment_hr_values = pulse_signal_processor.calculate_pulse_signals_for_segment(segment_data)
                    self._update_normalized_values_and_pulse_signals(segment_pulse_signals, segment_color_values,segment_snr_values, segment_hr_values, pulse_signals_by_region, normalized_values_by_region, snr_values_by_region, hr_values_by_region)
                    segment_data = self._retain_overlap_data(segment_data)

            video_progress.close() # Close the video progress bar
            mp_config.release_shared_counter(tqdm_position)

        # Ensure to close resources
        # self.face_detector.close()
        cap.release()
        cv2.destroyAllWindows()

        tracked_faces.clear()  # Clear the tracked faces dictionary
        segment_data.clear()  # Clear the segment data dictionary
        del pulse_signal_processor  # Delete the pulse signal processor
        gc.collect()  # Run garbage collection

        return normalized_values_by_region, pulse_signals_by_region, snr_values_by_region, hr_values_by_region, framerate

    def validate_masks(self, masks: Dict[str, Union[np.ndarray, List[np.ndarray]]]) -> bool:
        """
        Validates that no mask in the dictionary is an empty list or has a maximum value of 0.

        Args:
            masks (Dict[str, Union[np.ndarray, List[np.ndarray]]]): A dictionary of masks to validate.

        Returns:
            bool: True if all masks are valid, False otherwise.
        """
        for mask in masks.values():
            if isinstance(mask, np.ndarray):
                # Check if the mask is empty or has all zero values
                if mask.size == 0 or np.max(mask) == 0:
                    return False
            elif isinstance(mask, list):
                # Check if the list is empty or contains any empty or all-zero masks
                if len(mask) == 0 or all((patch.size == 0 or np.max(patch) == 0) for patch in mask if isinstance(patch, np.ndarray)):
                    return False
        return True
    
    def _update_normalized_values_and_pulse_signals(
        self,
        segment_pulse_signals: Dict[str, Any],
        segment_color_values: Dict[str, Any],
        segment_snr_values: Dict[str, Any],
        segment_hr_values: Dict[str, Any],
        pulse_signals: Dict[str, Any],
        normalized_color_values: Dict[str, Any],
        snr_values: Dict[str, Dict[str, Any]],
        hr_values: Dict[str, Dict[str, Any]],
    ):
        """
        Updates the normalized values and pulse signals with the calculated values for the current segment.

        Args:
            segment_pulse_signals (Dict[str, Dict[str, np.ndarray]]): Pulse signals for the current segment.
            segment_color_values (Dict[str, np.ndarray]): Color values for the current segment.
            segment_snr_values (Dict[str, Dict[str, float]]): Signal-to-noise ratio values for the current segment.
            segment_hr_values (Dict[str, Dict[str, float]]): Heart rate values for the current segment.
            pulse_signals (Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]): Accumulated pulse signals across all segments.
            normalized_color_values (Dict[str, Dict[str, List[np.ndarray]]]): Accumulated normalized color values across all segments.
            snr_values (Dict[str, Dict[str, List[float]]]): Accumulated SNR values across all segments.
            hr_values (Dict[str, Dict[str, List[float]]]): Accumulated HR values across all segments.
        """

        def append_to_dict_list(dictionary: Dict[str, Any], key: str, value):
            """
            Appends a value to a list in a dictionary. If the key does not exist, it creates a new list.

            Args:
                dictionary (Dict[str, Any]): The dictionary to append the value to.
                key (str): The key for the list in the dictionary.
                value: The value to append to the list.
            """
            if key not in dictionary:
                dictionary[key] = []
            dictionary[key].append(value)

        # Append data from segment dictionaries to accumulated dictionaries
        for region, values in segment_color_values.items():
            append_to_dict_list(normalized_color_values, region, values)

        for region, values in segment_snr_values.items():
            append_to_dict_list(snr_values, region, values)

        for region, values in segment_hr_values.items():
            append_to_dict_list(hr_values, region, values)

        for region, signals in segment_pulse_signals.items():
            append_to_dict_list(pulse_signals, region, signals)

         # Clear the segment dictionaries after usage
        segment_pulse_signals.clear()
        segment_color_values.clear()
        segment_snr_values.clear()
        segment_hr_values.clear()

    def _retain_overlap_data(
        self, segment_data: Dict[str, List[np.ndarray]]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Retains the overlap data for the next segment processing.

        Args:
            segment_data (Dict[str, List[np.ndarray]]): The data for the current segment.

        Returns:
            Dict[str, List[np.ndarray]]: The retained data for the next segment processing.
        """
        return {
            region: features[-self.overlap :]
            for region, features in segment_data.items()
        }

    def _visualize_frame(
        self,
        rgb_frame: np.ndarray,
        masks: Dict[str, np.ndarray],
        output_folder: str,
        frame_idx: int,
        process_type: str,
    ):
        """
        Visualizes the masks on the RGB frame and saves the visualized frames.

        Args:
            rgb_frame (np.ndarray): The RGB frame to overlay the masks on.
            masks (Dict[str, np.ndarray]): The masks to visualize.
            output_folder (str): The directory to save the visualized frames.
            frame_idx (int): The index of the frame being processed.
            process_type (str): The type of processing ('background' or 'patch').
        """
        os.makedirs(output_folder, exist_ok=True)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if process_type == "patch":
            for region_idx, (region, mask_list) in enumerate(masks.items()):
                for patch_idx, mask in enumerate(mask_list):
                    # Create a colored mask for each patch
                    colored_mask = np.zeros_like(bgr_frame)

                    # Ensure mask is broadcastable to 3D
                    if mask.ndim == 2:
                        mask = np.expand_dims(mask, axis=-1)

                    colored_mask[:, :, region_idx % 3] = mask[
                        :, :, 0
                    ]  # Cycle through R, G, B channels

                    # Overlay the colored mask on the original frame
                    visualized_frame = cv2.addWeighted(
                        bgr_frame, 0.7, colored_mask, 0.3, 0
                    )
                    output_path = os.path.join(
                        output_folder,
                        f"{region}_patch_{patch_idx}_frame_{frame_idx}.png",
                    )
                    cv2.imwrite(output_path, visualized_frame)
        else:
            for region, mask in masks.items():
                # Create a colored mask
                colored_mask = np.zeros_like(bgr_frame)
                colored_mask[:, :, 2] = mask  # Red channel

                # Overlay the colored mask on the original frame
                visualized_frame = cv2.addWeighted(bgr_frame, 0.7, colored_mask, 0.3, 0)
                output_path = os.path.join(
                    output_folder, f"{region}_mask_frame_{frame_idx}.png"
                )
                cv2.imwrite(output_path, visualized_frame)

