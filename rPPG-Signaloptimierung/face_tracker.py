"""
This module provides functionality for tracking faces in video frames. The primary class is FaceTracker, which includes methods for detecting, tracking, and drawing faces in video frames.

Classes:
    FaceTracker: A class used to track faces in video frames.
"""

import gc
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2
from rppg_facepatches.video_processor.face_detector import FaceDetector


class FaceTracker:
    """
    A class used to track faces in video frames. It supports detecting faces, identifying the closest face
    to a previously known key point, updating tracked faces, resetting tracking data, and drawing tracked faces
    on frames.

    Methods:
        detect_face(frame, last_kp_coord, frame_timestamp_ms, face_detector):
            Detects faces in a given frame and returns the closest face to the last known key point coordinates.

        find_closest_face(faces, last_kp_coord):
            Finds the face closest to the last known key point coordinates from a list of detected faces.

        update_tracked_faces(tracked_faces, next_face_id, face_rect, key_point_coord):
            Updates the dictionary of tracked faces with the current face and key point coordinates.

        reset_tracking(tracked_faces, segment_data, last_kp_coord=None):
            Resets the tracking information and clears the segment data.

        draw_tracked_faces(frame, tracked_faces):
            Draws rectangles and IDs around the tracked faces on the given frame.
    """

    def detect_face(
        self,
        frame: np.ndarray,
        last_kp_coord: Optional[Tuple[int, int]],
        face_detector: FaceDetector,
        frame_idx: Optional[int] = None,
    ) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray, Optional[List]]]:
        """
        Detects faces in a given frame and returns the closest face to the last known key point coordinates.

        Args:
            frame (np.ndarray): The RGB input frame in which to detect faces.
            last_kp_coord (Optional[Tuple[int, int]]): The last known coordinates of the key point.
            face_detector (FaceDetector): The face detector used to detect faces in the frame.
            frame_idx (Optional[int]): An optional frame index for saving visualizations.

        Returns:
        Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray, Optional[List]]]:
            A tuple containing:
                - Tuple[int, int, int, int]: The bounding box coordinates (x, y, width, height) of the detected face.
                - Tuple[int, int]: The coordinates (x, y) of a key point (e.g., nose tip) within the detected face.
                - np.ndarray: The cropped image region corresponding to the detected face.
                - Optional[List]: Any additional data associated with the detected face, such as facial landmarks.
            Returns None if no faces are detected in the frame.
        """
        faces = face_detector.detect_faces(frame, frame_idx=frame_idx)

        if faces == []:
            return None
        if last_kp_coord:
            return self.find_closest_face(faces, last_kp_coord)
        return faces[0]

    def find_closest_face(
        self,
        faces: List[Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray, Optional[List]]],
        last_kp_coord: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int], np.ndarray, Optional[List]]:
        """
        Finds the face closest to the last known key point coordinates from a list of detected faces.

        Args:
            faces (List[Tuple]): List of detected faces, each represented as a tuple containing the bounding box,
                                 key point coordinates, face region image, and optional additional data.
            last_kp_coord (Tuple[int, int]): The last known coordinates of the key point.

        Returns:
            Tuple: The face closest to the last known key point coordinates, represented as a tuple containing the
                   bounding box, key point coordinates, face region image, and optional additional data.
        """
        key_point_coords = np.array([kp_coord for _, kp_coord, _, _ in faces])
        distances = np.sum((key_point_coords - np.array(last_kp_coord)) ** 2, axis=1)
        closest_index = np.argmin(distances)
        return faces[closest_index]

    def update_tracked_faces(
        self,
        tracked_faces: Dict[int, Tuple[int, int, Tuple[int, int, int, int]]],
        next_face_id: int,
        face_rect: Tuple[int, int, int, int],
        key_point_coord: Tuple[int, int],
    ) -> Tuple[Dict[int, Tuple[int, int, Tuple[int, int, int, int]]], int]:
        """
        Updates the dictionary of tracked faces with the current face and key point coordinates.

        Args:
            tracked_faces (Dict[int, Tuple[int, int, Tuple[int, int, int, int]]]): Dictionary of tracked faces, where keys
                                                                                   are face IDs and values are tuples
                                                                                   containing key point coordinates and
                                                                                   bounding box.
            next_face_id (int): The next face ID to be assigned.
            face_rect (Tuple[int, int, int, int]): The bounding box of the face.
            key_point_coord (Tuple[int, int]): The coordinates of the key point.

        Returns:
            Tuple[Dict[int, Tuple[int, int, Tuple[int, int, int, int]]], int]: Updated dictionary of tracked faces and
                                                                               the next face ID to be used.
        """
        if tracked_faces:
            tracked_noses = np.array([(x, y) for x, y, _ in tracked_faces.values()])
            distances = np.sum((tracked_noses - np.array(key_point_coord)) ** 2, axis=1)
            min_distance = np.min(distances)
            if min_distance < 20000:
                closest_face_id = list(tracked_faces.keys())[np.argmin(distances)]
                tracked_faces[closest_face_id] = (*key_point_coord, face_rect)
            else:
                tracked_faces[next_face_id] = (*key_point_coord, face_rect)
                next_face_id += 1
        else:
            tracked_faces[next_face_id] = (*key_point_coord, face_rect)
            next_face_id += 1
        return tracked_faces, next_face_id

    def reset_tracking(
        self,
        tracked_faces: Dict[int, Tuple[int, int, Tuple[int, int, int, int]]],
        segment_data: Dict[str, List],
        last_kp_coord: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Resets the tracking information and clears the segment data.

        Args:
            tracked_faces (Dict[int, Tuple[int, int, Tuple[int, int, int, int]]]): Dictionary of tracked faces.
            segment_data (Dict[str, List]): Dictionary containing segment data to be cleared.
            last_kp_coord (Optional[Tuple[int, int]]): The last known coordinates of the key point, defaults to None.

        Returns:
            Optional[Tuple[int, int]]: None, indicating the last known key point coordinates have been reset.
        """
        tracked_faces.clear()
        last_kp_coord = None

        for region in segment_data.keys():
            segment_data[region].clear()

        gc.collect()

        return last_kp_coord

    def draw_tracked_faces(
        self,
        frame,
        tracked_faces: Dict[int, Tuple[int, int, Tuple[int, int, int, int]]],
    ):
        """
        Draws rectangles and IDs around the tracked faces on the given frame.

        Args:
            frame (np.ndarray): The frame on which to draw the rectangles and IDs.
            tracked_faces (Dict[int, Tuple[int, int, Tuple[int, int, int, int]]]): Dictionary of tracked faces, where keys
                                                                                   are face IDs and values are tuples
                                                                                   containing key point coordinates and
                                                                                   bounding box.
        """
        for face_id, (_, _, (x, y, w, h)) in tracked_faces.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"ID: {face_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
