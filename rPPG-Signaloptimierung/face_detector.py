"""
This module provides functionality for detecting faces and their landmarks in images using MediaPipe solutions. The primary class is FaceDetector, which includes methods for detecting faces, retrieving landmarks, and managing the models used for detection.

Classes:
    FaceDetector: A class used to process frames for face detection and landmark detection using MediaPipe solutions.
"""

import os
import logging
import warnings
from typing import List, Tuple, Optional
from pathlib import Path
import requests
import tqdm
import mediapipe as mp
import numpy as np
import cv2
from rppg_facepatches.video_processor import path_config
from rppg_facepatches.video_processor.mask_manager import MaskManager

# logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Disable oneDNN optimizations

# Set logging level to ERROR to suppress info/debug logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class FaceDetector:
    """
    A class used to process frames for face detection and landmark detection using MediaPipe solutions.

    Methods:
        close():
            Closes the face detector and face landmarks detector.

        detect_face(image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int]]]:
            Detects faces in the given image and returns bounding boxes and nose coordinates.
        detect_faces(frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray]]:
            Detects faces in the given image and returns bounding boxes, nose coordinates, and cropped face images.

        detect_face_landmarks(image: np.ndarray):
        detect_face_landmarks(frame: np.ndarray):
            Detects face landmarks in the given image.

        _get_bounding_box(relative_bounding_box, frame_cols, frame_rows) -> Tuple[int, int, int, int]:
            Calculates the bounding box coordinates from relative coordinates.

        _adjust_bounding_box(x: int, y: int, w: int, h: int, frame_cols: int, frame_rows: int) -> Tuple[int, int, int, int]:
            Adjusts the bounding box coordinates to ensure they do not exceed frame dimensions.

        _get_nose_coordinates(detection, frame_cols: int, frame_rows: int) -> Tuple[int, int]:
            Gets the nose coordinates from the detection.
    """

    def __init__(self):
        """
        Initializes the FaceDetector with face detection and face mesh models from MediaPipe.
        """
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7,
        )
        self.face_landmarks_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,  # Detect up to 5 faces
            refine_landmarks=True,  # Use the refined landmark model
        )

    def close(self):
        """
        Closes the face detector and face landmarks detector.
        """
        if self.face_detector:
            self.face_detector.close()
        if self.face_landmarks_detector:
            self.face_landmarks_detector.close()

    def detect_faces(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray, Optional[List]]]:
        """
        Detects faces in the given image using the specified face detection model.

        Args:
            face_detector (FaceDetection): The face detection model.
            image (ndarray): The input image.

        Returns:
            list: A list of tuples containing the bounding box and nose coordinates for each detected face.
        """
        detection_results = self.face_detector.process(frame)
        frame_rows, frame_cols, _ = frame.shape

        faces = []
        if detection_results.detections:
            # ih, iw, _ = image.shape
            for detection in detection_results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box

                # Calculate the bounding box coordinates
                x, y, w, h = self._get_bounding_box(
                    relative_bounding_box, frame_cols, frame_rows
                )

                # Ensure the bounding box dimensions are positive
                if w <= 0 or h <= 0:
                    print(f"Warning: Invalid bounding box dimensions at frame {frame_idx}: ({x}, {y}, {w}, {h})")
                    continue

                # Ensure the bounding box doesn't exceed image dimensions
                x, y, w, h = self._adjust_bounding_box(
                    x, y, w, h, frame_cols, frame_rows
                )

                # Check if the adjusted bounding box is valid
                if w <= 0 or h <= 0:
                    print(f"Warning: Adjusted bounding box is invalid at frame {frame_idx}: ({x}, {y}, {w}, {h})")
                    continue

                key_x, key_y = self._get_key_coordinates(
                    detection, frame_cols, frame_rows
                )

                # Crop the face region
                cropped_face = frame[y : y + h, x : x + w]

                # Check if the cropped face is valid
                if cropped_face.size == 0:
                    print(f"Warning: Cropped face has zero size at frame {frame_idx}: ({x}, {y}, {w}, {h})")
                    continue

                landmarks = self.detect_face_landmarks(cropped_face, frame_idx)

                # Append bounding box, nose coordinates, and cropped face to the faces list
                faces.append(((x, y, w, h), (key_x, key_y), cropped_face, landmarks))

        return faces

    def detect_face_landmarks(self, frame: np.ndarray, frame_idx: Optional[int]):
        """
        Detects face landmarks in the given image.

        Args:
            detector (FaceMesh): The face landmarks detector.
            image (ndarray): The input image.

        Returns:
            list: The detected face landmarks, or None if no landmarks are found.
        """
        results = self.face_landmarks_detector.process(frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # self.visualize_landmarks_on_cropped_face(frame, landmarks, frame_idx)
            return landmarks
        return None

    def _get_bounding_box(
        self, relative_bounding_box, frame_cols, frame_rows
    ) -> Tuple[int, int, int, int]:
        """
        Calculates the bounding box coordinates from relative coordinates.

        Args:
            relative_bounding_box: The relative bounding box coordinates.
            frame_cols (int): The number of columns in the frame.
            frame_rows (int): The number of rows in the frame.

        Returns:
            Tuple[int, int, int, int]: The bounding box coordinates.
        """
        x = int(relative_bounding_box.xmin * frame_cols)
        y = int(relative_bounding_box.ymin * frame_rows)
        w = int(relative_bounding_box.width * frame_cols)
        h = int(relative_bounding_box.height * frame_rows)

        # Add padding
        padding = int(max(w, h) * 0.3)
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        return x, y, w, h

    def _adjust_bounding_box(
        self, x: int, y: int, w: int, h: int, frame_cols: int, frame_rows: int
    ) -> Tuple[int, int, int, int]:
        """
        Adjusts the bounding box coordinates to ensure they do not exceed frame dimensions.

        Args:
            x (int): The x-coordinate of the bounding box.
            y (int): The y-coordinate of the bounding box.
            w (int): The width of the bounding box.
            h (int): The height of the bounding box.
            frame_cols (int): The number of columns in the frame.
            frame_rows (int): The number of rows in the frame.

        Returns:
            Tuple[int, int, int, int]: The adjusted bounding box coordinates.
        """
        if x < 0:
            w += x  # Reduce width by the amount of the out-of-bounds x-coordinate
            x = 0
        if y < 0:
            h += y  # Reduce height by the amount of the out-of-bounds y-coordinate
            y = 0
        if x + w > frame_cols:
            w = frame_cols - x
        if y + h > frame_rows:
            h = frame_rows - y
        return x, y, max(0, w), max(0, h)

    def _get_key_coordinates(
        self, detection, frame_cols: int, frame_rows: int
    ) -> Tuple[int, int]:
        """
        Gets the nose coordinates from the detection.

        Args:
            detection: The detection result.
            frame_cols (int): The number of columns in the frame.
            frame_rows (int): The number of rows in the frame.

        Returns:
            Tuple[int, int]: The nose coordinates.
        """
        key_point = mp.solutions.face_detection.get_key_point(
            detection, mp.solutions.face_detection.FaceKeyPoint.NOSE_TIP
        )
        key_point_x = int(key_point.x * frame_cols)
        key_point_y = int(key_point.y * frame_rows)
        return key_point_x, key_point_y

    def visualize_landmarks_on_cropped_face(self, original_cropped_face: np.ndarray, landmarks, frame_idx: Optional[int] = None):
        """
        Visualizes landmarks on the cropped face.

        Args:
            cropped_face (np.ndarray): The cropped face image.
            landmarks: The landmarks detected on the face.
            frame_idx (Optional[int]): Index of the current frame for unique output filename generation.
        """
        # Make a copy of the original cropped face to draw landmarks
        cropped_face = original_cropped_face.copy()

        if landmarks is not None:
            for landmark in landmarks:
                x = int(landmark.x * cropped_face.shape[1])
                y = int(landmark.y * cropped_face.shape[0])
                # Draw landmarks as blue dots
                cv2.circle(cropped_face, (x, y), radius=2, color=(255, 0, 0), thickness=-1)

        # Calculate and draw the face center
        # x_center, y_center = self.get_face_center(landmarks)
        # x_center = int(x_center * cropped_face.shape[1])
        # y_center = int(y_center * cropped_face.shape[0])
        # cv2.circle(cropped_face, (x_center, y_center), radius=5, color=(0, 255, 0), thickness=-1)

        # Convert the frame from RGB to BGR
        bgr_cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

        # Display the cropped face with landmarks
        # cv2.imshow("Cropped Face with Landmarks", bgr_cropped_face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Save the image with landmarks
        if frame_idx is None:
            filename = 'cropped_face_landmarks.png'
        else:
            filename = f'{frame_idx}_cropped_face_landmarks.png'

        output_path = path_config.PROJECT_PATH /'cropped_faces' / path_config.VIDEO_FILE_PATH.stem
        output_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path / filename), bgr_cropped_face)


class FaceDetectorNew:
    """
    A class for processing frames to detect faces and landmarks using MediaPipe solutions.

    Methods:
        close():
            Closes the face detector and landmarks detector.

        detect_faces(frame: np.ndarray, frame_timestamp_ms: int, confidence_threshold: float) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray, Optional[List]]]:
            Detects faces in the given image and returns bounding boxes, nose coordinates, cropped face images, and landmarks.

        visualize_keypoints(frame: np.ndarray, keypoints, special_index: int, output_filename: str) -> None:
            Visualizes and highlights keypoints on the frame.

        download_models_if_necessary():
            Downloads models if they are not already present.

        check_weight(model_path: Path, model_url: str) -> bool:
            Checks if the weight file exists, downloads it if not.

        download_weight(url: str, target_path: Path):
            Downloads the weight file from the given URL to the target path.

        init_face_detector():
            Initializes the face detector with specified options.

        init_face_landmarker():
            Initializes the face landmarker with specified options.
    """

    def __init__(self):
        """
        Initializes the FaceDetector with face detection and face mesh models from MediaPipe.
        """
        self.mask_manager = MaskManager()

        # Define paths and URLs for models
        self.weights_folder = Path(__file__).resolve().parent.parent / "weights"
        self.weights_folder.mkdir(exist_ok=True)

        self.face_detector_model_path = self.weights_folder / "blaze_face_short_range.tflite"
        self.face_detector_model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
        )
        
        self.face_landmarker_model_path = self.weights_folder / "face_landmarker.task"
        self.face_landmarker_model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )

        # Download models if they do not exist
        self.download_models_if_necessary()
        
        # Initialize MediaPipe tasks with downloaded models
        self.init_face_detector()
        self.init_face_landmarker()

    def init_face_detector(self):
        """Initialize face detector."""
        face_detector_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.face_detector_model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.5
        )
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(face_detector_options)

    def init_face_landmarker(self):
        """Initialize face landmarker."""
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.face_landmarker_model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=3,
            # min_tracking_confidence=0.5,
            # min_face_detection_confidence=0.5,
            # min_face_presence_confidence=0.5,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_landmarker_options)

    def download_models_if_necessary(self):
        """Download models if they are not already present."""
        if not self.check_weight(self.face_detector_model_path, self.face_detector_model_url):
            raise RuntimeError("Failed to fetch model for face detection.")
        if not self.check_weight(self.face_landmarker_model_path, self.face_landmarker_model_url):
            raise RuntimeError("Failed to fetch model for face mesh.")
        
    def check_weight(self, model_path: Path, model_url: str) -> bool:
        """
        Check if the model weight file is present; download it if absent.

        Args:
            model_path (Path): Path to the model file.
            model_url (str): URL to download the model if not present.

        Returns:
            bool: True if the model file exists or is successfully downloaded, False otherwise.
        """
        if not model_path.exists():
            self.download_weight(model_url, model_path)
        return model_path.exists()

    def download_weight(self, url: str, target_path: Path):
        """
        Download the weight file from the specified URL to the target path.

        Args:
            url (str): URL from which to download the weight file.
            target_path (Path): Path to save the downloaded weight file.
        """
        target_path.parent.mkdir(exist_ok=True, parents=True)
        response = requests.get(url, timeout=3, stream=True)
        assert response.status_code == 200, 'Failed to download weights'

        def save_response_content(response, destination):
            chunk_size = 32768
            total = int(response.headers.get('content-length', 0))
            with open(destination, 'wb') as f, tqdm.tqdm(
                desc=destination,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size):
                    if chunk:
                        size = f.write(chunk)
                        progress_bar.update(size)

        print(f'Downloading weights {target_path.stem}')
        save_response_content(response, str(target_path))
        print('Download finished.')

    def close(self):
        """
        Closes the face detector and face landmarks detector.
        """
        if self.face_detector:
            self.face_detector.close()
        if self.face_landmarker:
            self.face_landmarker.close()

    def detect_faces(self, frame: np.ndarray, frame_timestamp_ms: int, confidence_threshold: float = 0.5, frame_idx: Optional[int] = None) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray, Optional[List]]]:
        """
        Detects faces in the given image and returns bounding boxes, nose coordinates, cropped face images, and landmarks.

        Args:
            frame (np.ndarray): The input RGB frame.
            frame_timestamp_ms (int): The timestamp of the frame in milliseconds.
            confidence_threshold (float): The minimum score required to consider a detection valid.

        Returns:
            List[Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray, Optional[List]]]: 
            A list of tuples containing the bounding box, nose coordinates, cropped face, and landmarks for each detected face.
        """
        timestamp = frame_timestamp_ms
        # Convert the frame to a MediaPipe Image
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect faces
        face_detector_result = self.face_detector.detect_for_video(mp_frame, timestamp)

        faces = []
        landmarks = None
        required_indices = self.mask_manager._left_cheek + self.mask_manager._right_cheek + self.mask_manager._forehead

        if face_detector_result.detections:
            for i, detection in enumerate(face_detector_result.detections):
                detection_score = detection.categories[0].score
                if detection_score >= confidence_threshold:
                    # Calculate a new centered bounding box with padding
                    bbox = detection.bounding_box
                    x, y, w, h = bbox.origin_x, bbox.origin_y , bbox.width, bbox.height
                    y = y + 2*h
                    # w = w*2
                    # h = h*2
                    bounding_box =  x, y, w, h

                    # Crop the face region
                    cropped_face_np = frame[y:y+h, x:x+w]
                    cropped_face = cropped_face_np.astype(np.uint8)

                    cropped_mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_face)
                    landmarker_results = self.face_landmarker.detect(cropped_mp_frame)

                    # Extract landmarks for this detection
                    if i < len(landmarker_results.face_landmarks):
                        landmarks = landmarker_results.face_landmarks[i]

                        if self.check_specific_landmarks(landmarks, required_indices):

                            # Assume nose tip is at index 1
                            kp_landmark = landmarks[1]  # Use the appropriate index for the nose tip
                            kp_x = int(kp_landmark.x * cropped_face.shape[1])
                            kp_y = int(kp_landmark.y * cropped_face.shape[0])

                            self.visualize_landmarks_on_cropped_face(cropped_face, landmarks, frame_idx)


                    # Append bounding box, key point coordinate, and cropped face to the faces list
                    faces.append((bounding_box, (kp_x, kp_y), cropped_face, landmarks))

        return faces

    def check_specific_landmarks(self, landmarks, required_indices):
        if not landmarks:
            return False

        found_indices = set()

        for idx, landmark in enumerate(landmarks):
            if idx in required_indices:
                found_indices.add(idx)

        return all(index in found_indices for index in required_indices)

    def visualize_landmarks_on_cropped_face(self, original_cropped_face: np.ndarray, landmarks, frame_idx: Optional[int] = None):
        """
        Visualizes landmarks on the cropped face.

        Args:
            cropped_face (np.ndarray): The cropped face image.
            landmarks: The landmarks detected on the face.
            frame_idx (Optional[int]): Index of the current frame for unique output filename generation.
        """
        # Make a copy of the original cropped face to draw landmarks
        cropped_face = original_cropped_face.copy()

        if landmarks is not None:
            for landmark in landmarks:
                x = int(landmark.x * cropped_face.shape[1])
                y = int(landmark.y * cropped_face.shape[0])
                # Draw landmarks as blue dots
                cv2.circle(cropped_face, (x, y), radius=2, color=(255, 0, 0), thickness=-1)

        # Calculate and draw the face center
        # x_center, y_center = self.get_face_center(landmarks)
        # x_center = int(x_center * cropped_face.shape[1])
        # y_center = int(y_center * cropped_face.shape[0])
        # cv2.circle(cropped_face, (x_center, y_center), radius=5, color=(0, 255, 0), thickness=-1)

        # Convert the frame from RGB to BGR
        bgr_cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

        # Display the cropped face with landmarks
        # cv2.imshow("Cropped Face with Landmarks", bgr_cropped_face)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Save the image with landmarks
        if frame_idx is None:
            filename = 'cropped_face_landmarks.png'
        else:
            filename = f'{frame_idx}_cropped_face_landmarks.png'

        output_path = path_config.PROJECT_PATH /'cropped_faces' / path_config.VIDEO_FILE_PATH.stem
        output_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path / filename), bgr_cropped_face)

    def visualize_keypoints(self, original_frame: np.ndarray, keypoints, special_index: int, frame_idx: Optional[int] = None) -> None:
        """
        Visualizes and highlights keypoints on the frame.

        Args:
            frame (np.ndarray): The original rgb frame to draw keypoints on.
            keypoints: The keypoints detected by the face detection model.
            special_index (int): The index of the keypoint to be highlighted differently.
            output_filename (str): The filename to save the image with visualized keypoints.
        """
        frame = original_frame.copy()

        for idx, keypoint in enumerate(keypoints):
            x = int(keypoint.x * frame.shape[1])
            y = int(keypoint.y * frame.shape[0])
            
            if idx == special_index:
                # Highlight the specific keypoint with a different color (red)
                cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            else:
                # Draw other keypoints in green
                cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # Convert the frame from RGB to BGR
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the image with keypoints
        # cv2.imshow("Keypoints Visualization", bgr_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        output_path = path_config.PROJECT_PATH / 'keypoint_visualization' / path_config.VIDEO_FILE_PATH.stem

         # Ensure the output path exists
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # Visualize keypoints on the frame
        if frame_idx is None:
            filename = 'keypoints_visualization.png'
        else:
            filename = f'{frame_idx}_keypoints_visualization.png'

        # Save the image with keypoints
        cv2.imwrite(str(output_path / filename), bgr_frame)


if __name__ == "__main__":
    detector = FaceDetector()
    detector.close()