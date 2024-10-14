"""
This module provides functionality for managing and creating masks for facial regions in images. The primary class is MaskManager, which includes methods for creating masks for different facial regions, splitting masks into patches, and calculating coordinates of mask elements.

Classes:
    MaskManager: A class used to manage and create masks for facial regions in images.
"""

from typing import List, Tuple, Dict, Union, Optional
from datetime import datetime
import random
import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from rppg_facepatches.video_processor import path_config


class MaskManager:
    """
    A class used to manage and create masks for facial regions in images. The masks can be for specific facial regions
    like cheeks and forehead, or for the background of the image. Additionally, masks can be split into patches
    for further analysis or processing.

    Attributes:
        num_patches (int): The number of patches to split each mask into.
        random_state (int): The seed for random number generation to ensure reproducibility.
        _left_cheek (List[int]): Landmark indices for the left cheek region.
        _right_cheek (List[int]): Landmark indices for the right cheek region.
        _forehead (List[int]): Landmark indices for the forehead region.

    Methods:
        create_masks(frame: np.ndarray, face_landmarks, face_rect: Tuple[int, int, int, int] = (0, 0, 1, 1)) -> Tuple[Dict[str, Union[np.ndarray, List[np.ndarray]]], np.ndarray]:
            Creates masks for specified facial regions or the background based on the type of process defined in path_config.

        _create_facial_masks(frame: np.ndarray, face_landmarks: List[NormalizedLandmark]) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
            Creates masks for facial regions (left cheek, right cheek, and forehead) and optionally splits them into patches.

        _create_facial_mask(frame: np.ndarray, face_landmarks: Optional[List[NormalizedLandmark]], face_rect: Optional[Tuple[int, int, int, int]], region_landmarks: Optional[List[int]] = None) -> np.ndarray:
            Creates a mask for a specified facial region using a convex hull on the frame.

        _create_rectangular_face_mask(mask: np.ndarray, face_rect: Optional[Tuple[int, int, int, int]]) -> None:
            Creates a rectangular mask based on the provided face rectangle.

        _create_landmark_based_mask(frame: np.ndarray, mask: np.ndarray, face_landmarks: List[NormalizedLandmark], face_rect: Optional[Tuple[int, int, int, int]], region_landmarks: Optional[List[int]]) -> None:
            Creates a mask using facial landmarks by constructing a convex hull over the specified region.

        _get_landmarks_list(face_landmarks: List[NormalizedLandmark], width: int, height: int, offset_x: int, offset_y: int, region_landmarks: Optional[List[int]]) -> List[Tuple[int, int]]:
            Retrieves the list of landmark positions based on given indices.

        save_masked_image(frame: np.ndarray, mask: np.ndarray, filename: str) -> None:
            Saves an image of the frame with the mask applied and overlayed in color.

        _create_background_mask(frame: np.ndarray, face_landmarks, face_rect: Tuple[int, int, int, int], height: int = 128, width: int = 128, margin: int = 10) -> Tuple[np.ndarray, np.ndarray]:
            Creates a background mask with a specified size and margin, avoiding overlap with the facial region.

        _split_masks_into_patches(masks: Dict[str, np.ndarray], method: str = 'fixed') -> Dict[str, List[np.ndarray]]:
            Splits each mask into patches using the specified method (fixed or random).

        _split_mask_into_random_patches(mask: np.ndarray) -> List[np.ndarray]:
            Splits a mask into random patches.

        _split_mask_into_fixed_patches(mask: np.ndarray) -> List[np.ndarray]:
            Splits a mask into fixed patches using clustering.

        _calculate_patch_coordinates(masks_patches: List[np.ndarray]) -> List[List[Tuple[int, int]]]:
            Calculates the coordinates of nonzero elements for each patch mask.

        _calculate_mask_coordinates(mask: np.ndarray) -> List[Tuple[int, int]]:
            Calculates the coordinates of nonzero elements in a mask.
    """

    def __init__(self, num_patches: int = 10, grid_size: int = 50, random_state: int = 42):
        """
        Initializes the MaskManager class.

        Args:
            num_patches (int, optional): Number of patches to split each mask into. Defaults to 10.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.num_patches = num_patches
        self.random_state = random_state
        
        self.grid_size = grid_size
        self.centroids: Optional[np.ndarray] = None # To store the current centroids by region
        self.centroids_history: Dict[str, List[np.ndarray]] = {} # To store historical centroids by region
        self.bounding_boxes: Dict[str, List[Tuple[int, int, int, int]]]
        self.weights = {
            'grid_index': 0.4,
            'relative_distances': 0.6,
            'sub_grid_index': 0.4,
            'proximity_to_grid_lines': 0.2,
            'distance_to_center': 0.55,
        }

        self._left_cheek = [
            147, 213, 192, 207, 205, 36, 142, 126,
            47, 121, 120, 119, 118, 117, 123,
        ]
        self._left_cheek_patches = [[147, 187, 207, 192, 213],
                                    [123, 50, 187, 147],
                                    [50, 205, 207, 187],
                                    [117, 118, 50, 123],
                                    [118, 36, 205, 50],
                                    [118, 119, 120, 100, 142, 36],
                                    [120, 121, 47, 126, 142, 100]]

        self._right_cheek = [
            355, 277, 350, 349, 348, 347, 346, 352,
            376, 433, 416, 427, 425, 266, 371,
        ]
        self._right_cheek_patches = [[427, 411, 376, 433, 416],
                                     [425, 280, 411, 427],
                                     [280, 352, 376, 411],
                                     [347, 346, 352, 280],
                                     [280, 425, 266, 347],
                                     [266, 371, 329, 349, 348, 347],
                                     [371, 355, 277, 350, 349, 329]]

        self._forehead = [338, 10, 109, 69, 66, 107, 8, 336, 296, 299]
        self._forehead_patches = [[69, 109, 108],
                                  [109, 10, 151, 108],
                                  [10, 338, 337, 151],
                                  [338, 299, 337],
                                  [337, 299, 296, 336],
                                  [151, 337, 336, 8],
                                  [108, 151, 8, 107],
                                  [69, 108, 107, 66]]

    def create_masks(self, frame: np.ndarray, face_landmarks, face_rect: Tuple[int, int, int, int] = (0, 0, 1, 1), frame_idx: Optional[int] = None) -> Tuple[Dict[str, Union[np.ndarray, List[np.ndarray]]], np.ndarray]:
        """
        Creates masks for specified facial regions or background.

        Args:
            frame (np.ndarray): The input frame containing the face.
            face_landmarks: The detected face landmarks.
            process_type (str): The type of process ('mask', 'patch', or 'background').
            face_rect (Tuple[int, int, int, int], optional): The bounding box of the face. Defaults to (0, 0, 1, 1).

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]: A dictionary of masks and the processed frame.
        """
        masks: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {}

        if path_config.ROI_TYPE in ["mask", "patch"]:
            masks = self._create_facial_masks(frame, face_landmarks, frame_idx)

        elif path_config.ROI_TYPE == "background":
            background_mask, cropped_frame = self._create_background_mask(frame, face_landmarks, face_rect)
            masks = {"background": background_mask}
            frame = cropped_frame

        else:
            raise ValueError(f"Invalid process_type: {path_config.ROI_TYPE}")

        return masks, frame
  
    def _create_facial_masks(self, frame: np.ndarray, face_landmarks: List[NormalizedLandmark], frame_idx: Optional[int] = None) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Creates masks for facial regions (left cheek, right cheek, and forehead).

        Args:
            frame (np.ndarray): The input frame containing the face.
            face_landmarks: The detected face landmarks.

        Returns:
            Dict[str, Union[np.ndarray, List[np.ndarray]]]: A dictionary of masks for each facial region.
        """
        if path_config.ROI_TYPE == "patch":
            # Split each region into patches
            left_cheek_patches = self._create_facial_patch_masks(frame, face_landmarks, self._left_cheek_patches)
            # self.save_masked_image(frame, left_cheek_patches, 'left_cheek', frame_idx)
            right_cheek_patches = self._create_facial_patch_masks(frame, face_landmarks, self._right_cheek_patches)
            # self.save_masked_image(frame, right_cheek_patches, 'right_cheek', frame_idx)
            forehead_patches = self._create_facial_patch_masks(frame, face_landmarks, self._forehead_patches)
            # self.save_masked_image(frame, forehead_patches, 'forehead', frame_idx)

            # Combine patches into full face and cheek masks
            cheeks_patches = left_cheek_patches + right_cheek_patches
            # self.save_masked_image(frame, cheeks_patches, 'cheeks', frame_idx)
            full_face_patches = cheeks_patches + forehead_patches
            self.save_masked_image(frame, full_face_patches, 'full_face', frame_idx)

            masks = {
                "forehead": forehead_patches,
                "left_cheek": left_cheek_patches,
                "right_cheek": right_cheek_patches,
                "cheeks": cheeks_patches,
                "full_face": full_face_patches,
            }

        else:
            # Create masks for individual regions
            left_cheek_mask = self._create_facial_mask(frame, face_landmarks, region_landmarks=self._left_cheek)
            # self.save_masked_image(frame, left_cheek_mask, 'left_cheek', frame_idx)
            right_cheek_mask = self._create_facial_mask(frame, face_landmarks, region_landmarks=self._right_cheek)
            # self.save_masked_image(frame, right_cheek_mask, 'right_cheek', frame_idx)
            forehead_mask = self._create_facial_mask(frame, face_landmarks, region_landmarks=self._forehead)
            # self.save_masked_image(frame, forehead_mask, 'forehead', frame_idx)

            # Combine individual masks into cheeks and full face masks
            cheeks_mask = left_cheek_mask | right_cheek_mask
            # self.save_masked_image(frame, cheeks_mask, 'cheeks', frame_idx)

            full_face_mask = cheeks_mask | forehead_mask
            # self.save_masked_image(frame, full_face_mask, 'full_face', frame_idx)

            masks = {
                "forehead": forehead_mask,
                "left_cheek": left_cheek_mask,
                "right_cheek": right_cheek_mask,
                "cheeks": cheeks_mask,
                "full_face": full_face_mask,
            }

        return masks
    
    def _create_facial_patch_masks(
        self,
        frame: np.ndarray,
        face_landmarks: List[NormalizedLandmark],
        region_landmarks: List[List[int]],
    ) -> List[np.ndarray]:
        """
        Creates multiple masks for a specified facial region using convex hull on the frame,
        dividing the region into smaller patches.

        Args:
            frame (np.ndarray): The RGB frame containing the face.
            face_landmarks (Optional[List[NormalizedLandmark]]): The detected face landmarks.
            face_rect (Optional[Tuple[int, int, int, int]]): The bounding box of the face (x, y, width, height).
            region_landmarks (Optional[List[int]]): List of landmark indices representing the facial region. 
                                                    If `None`, all landmarks will be used.

        Raises:
            ValueError: If no landmarks are found for the specified region.

        Returns:
            List[np.ndarray]: A list of masks, each representing a patch of the specified facial region.
        """
        masks = [np.zeros(frame.shape[:2], dtype=np.uint8) for _ in region_landmarks]

        for idx, patch_landmarks in enumerate(region_landmarks):
            self._create_landmark_based_patch_mask(frame, masks[idx], face_landmarks, patch_landmarks)

        return masks

    def _create_landmark_based_patch_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        face_landmarks: List[NormalizedLandmark],
        patch_landmarks: List[int],
    ):
        """
        Create a mask using face landmarks for a specific patch.

        Args:
            frame (np.ndarray): The RGB frame containing the face.
            mask (np.ndarray): The mask to be modified.
            face_landmarks (List[NormalizedLandmark]): The detected face landmarks.
            patch_landmarks (List[int]): List of landmark indices representing the patch.
        """
        h, w = frame.shape[:2]

        # Convert the patch landmarks to coordinates
        landmarks_list = self._get_landmarks_list(face_landmarks, w, h, 0, 0, patch_landmarks)

        if not landmarks_list:
            raise ValueError("No landmarks found in the specified region.")

        landmarks_array = np.array(landmarks_list, dtype=np.int32)
        hull = cv2.convexHull(landmarks_array)
        
        # Fill the mask using the convex hull of the patch
        cv2.fillConvexPoly(img=mask, points=hull, color=[float(255)])

    def _create_facial_mask(
            self,
            frame: np.ndarray,
            face_landmarks: Optional[List[NormalizedLandmark]],
            face_rect: Optional[Tuple[int, int, int, int]] = None,
            region_landmarks: Optional[List[int]] = None
        ) -> np.ndarray:
        """
        Creates a mask for a specified facial region using convex hull on the frame.

        Args:
            frame (np.ndarray): The RGB frame containing the face.
            face_landmarks (Optional[List[NormalizedLandmark]]): The detected face landmarks.
            face_rect (Optional[Tuple[int, int, int, int]]): The bounding box of the face (x, y, width, height).
            region_landmarks (Optional[List[int]]): List of landmark indices representing the facial region. 
                                                    If `None`, all landmarks will be used.

        Raises:
            ValueError: If no landmarks are found for the specified region.

        Returns:
            np.ndarray: The mask representing the specified facial region.
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if face_landmarks is None:
            self._create_rectangular_face_mask(mask, face_rect)

        else:
            self._create_landmark_based_mask(frame, mask, face_landmarks, face_rect, region_landmarks)

        # self.save_masked_image(frame, mask, 'masked_face')

        return mask

    def _create_rectangular_face_mask(self, mask: np.ndarray, face_rect: Optional[Tuple[int, int, int, int]]):
        """
        Create a rectangular mask based on face_rect.

        Args:
            mask (np.ndarray): The mask to be modified.
            face_rect (Optional[Tuple[int, int, int, int]]): The bounding box of the face (x, y, width, height).
        """
        if face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(mask, (x, y), (x + w, y + h), color=[float(255)], thickness=-1)
        else:
            mask

    def _create_landmark_based_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        face_landmarks: List[NormalizedLandmark],
        face_rect: Optional[Tuple[int, int, int, int]],
        region_landmarks: Optional[List[int]]
    ):
        """
        Create a mask using face landmarks.

        Args:
            frame (np.ndarray): The RGB frame containing the face.
            mask (np.ndarray): The mask to be modified.
            face_landmarks (List[NormalizedLandmark]): The detected face landmarks.
            face_rect (Optional[Tuple[int, int, int, int]]): The bounding box of the face (x, y, width, height).
            region_landmarks (Optional[List[int]]): List of landmark indices representing the facial region. 
                                                    If `None`, all landmarks will be used.
        """
        if face_rect is None:
            h, w = frame.shape[:2]
            x, y = 0, 0
        else:
            x, y, w, h = face_rect

        landmarks_list = self._get_landmarks_list(face_landmarks, w, h, x, y, region_landmarks)

        if not landmarks_list:
            raise ValueError("No landmarks found in the specified region.")

        landmarks_array = np.array(landmarks_list, dtype=np.int32)
        hull = cv2.convexHull(landmarks_array)
        cv2.fillConvexPoly(img=mask, points=hull, color=[float(255)])

    def _get_landmarks_list(
        self,
        face_landmarks: List[NormalizedLandmark],
        width: int,
        height: int,
        offset_x: int,
        offset_y: int,
        region_landmarks: Optional[List[int]]
    ) -> List[Tuple[int, int]]:
        """
        Get the list of landmarks positions.

        Args:
            face_landmarks (List[NormalizedLandmark]): The detected face landmarks.
            width (int): Width of the frame or face bounding box.
            height (int): Height of the frame or face bounding box.
            offset_x (int): X offset for landmarks.
            offset_y (int): Y offset for landmarks.
            region_landmarks (Optional[List[int]]): List of landmark indices representing the facial region.

        Returns:
            List[Tuple[int, int]]: List of (x, y) positions for the landmarks.
        """
        if region_landmarks is None:
            return [
                (int(landmark.x * width + offset_x), int(landmark.y * height + offset_y))
                for landmark in face_landmarks
            ]

        return [
            (int(landmark.x * width + offset_x), int(landmark.y * height + offset_y))
            for idx, landmark in enumerate(face_landmarks)
            if idx in region_landmarks
        ]
   
    def _create_background_mask(self, frame: np.ndarray, face_landmarks, face_rect: Tuple[int, int, int, int], height: int = 128, width: int = 128, margin: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a background mask with a specified size and margin.

        Args:
            frame (np.ndarray): The input frame.
            face_landmarks: The detected face landmarks.
            face_rect (Tuple[int, int, int, int]): The bounding box of the face.
            height (int, optional): The height of the background mask. Defaults to 128.
            width (int, optional): The width of the background mask. Defaults to 128.
            margin (int, optional): The margin around the mask. Defaults to 10.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The background mask and the cropped frame.
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Determine starting position
        start_x, start_y = (margin, margin)
        if face_rect != (0, 0, 1, 1):
            face_mask = self._create_facial_mask(frame, face_landmarks, face_rect=face_rect)

            # Function to check mask overlap
            def is_mask_overlapping(face_mask, x, y, width, height):
                mask_temp = np.zeros_like(face_mask)
                mask_temp[y : y + height, x : x + width] = 255
                return np.any(cv2.bitwise_and(face_mask, mask_temp))

            # Search for a non-overlapping position
            while is_mask_overlapping(face_mask, start_x, start_y, width, height):
                start_x += 1
                if start_x + width + margin > frame.shape[1]:
                    start_x = margin
                    start_y += 1
                if start_y + height + margin > frame.shape[0]:
                    raise ValueError("Cannot place mask without overlapping the face.")

        # Create mask and cropped frame
        mask[start_y : start_y + height, start_x : start_x + width] = 255
        cropped_frame = frame[start_y : start_y + height, start_x : start_x + width]
        cropped_mask = mask[start_y : start_y + height, start_x : start_x + width]

        self.save_masked_image(frame, mask, 'masked_background.png')
        
        return cropped_mask, cropped_frame

    def _calculate_mask_coordinates(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Calculate the coordinates of nonzero elements in a mask.

        Args:
            mask (np.ndarray): A mask represented as a NumPy array.

        Returns:
            List[Tuple[int, int]]: List of tuples representing coordinates of nonzero elements.
        """
        # Get coordinates of nonzero elements in the mask
        mask_coords = np.argwhere(mask == 255)

        # Convert ndarray to list of tuples using list comprehension
        return [(int(coord[0]), int(coord[1])) for coord in mask_coords]

    def save_masked_image(self, frame: np.ndarray, masks: Union[np.ndarray, List[np.ndarray]], filename: str, frame_idx: Optional[int] = None) -> None:
        """
        Saves an image of the frame with multiple masks applied and overlayed in different colors.

        Args:
            cropped_frame (np.ndarray): The cropped frame containing the face.
            mask (Union[np.ndarray, List[np.ndarray]]): The mask to apply to the frame.
            filename (str): The file name to save the image as.
        """
        # Create a copy of the frame to overlay the colored patches
        colored_frame = frame.copy()

        # If a single mask array is provided, convert it into a list
        if isinstance(masks, np.ndarray):
            masks = [masks]

        # Ensure unique colors for each patch without repetition
        distinct_colors = [
            (106, 90, 205), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (255, 128, 0), (128, 0, 128), (128, 64, 0), (255, 192, 203),
            (128, 128, 0), (0, 128, 128), (128, 255, 0), (64, 224, 208), (75, 0, 130),
            (255, 215, 0), (250, 128, 114), (255, 127, 80), (135, 206, 235), (238, 130, 238),
            (189, 252, 201)
        ] # List of predefined distinct colors
        random.shuffle(distinct_colors)  # Shuffle to randomize color assignment

        patch_colors = {}

        # Iterate over each mask and apply a distinct color
        for i, (mask, distinct_color) in enumerate(zip(masks, distinct_colors)):
                if i == 12:
                    if isinstance(masks, np.ndarray):
                        color = (0, 0, 255)  # Red
                    else:
                        # color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Random color
                        color = distinct_color
                    color = (255, 0, 0)  # Red
                    colored_frame[mask == 255] = color
                    patch_colors[i] = color  # Store the color for each patch

        # # Display the overlayed frame for debugging
        # cv2.imshow("Frame", colored_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add timestamp to the filename
        if frame_idx is not None:
            legend_filename = f"{filename}_{path_config.ROI_TYPE}_{path_config.VIDEO_INDEX}_{path_config.VIDEO_FILE_PATH.stem}_frame{frame_idx}_legend_{timestamp}.png"
            filename = f"{filename}_{path_config.ROI_TYPE}_{path_config.VIDEO_INDEX}_{path_config.VIDEO_FILE_PATH.stem}_frame{frame_idx}_{timestamp}.png"
        else:
            legend_filename = f"{filename}_legend_{timestamp}.png"
            filename = f"{filename}_{path_config.ROI_TYPE}_{timestamp}.png"

        # Construct the image path
        output_path = path_config.MASK_VISUALIZER_OUTPUT_DIR
        output_path.mkdir(parents=True, exist_ok=True)

        # Define full paths for the overlay and legend images
        overlay_output_path = output_path / filename
        legend_output_path = output_path / legend_filename

        # Convert BGR frame back to RGB before saving
        rgb_colored_frame = cv2.cvtColor(colored_frame, cv2.COLOR_BGR2RGB)

        # Save the overlayed frame to a file
        cv2.imwrite(str(overlay_output_path), rgb_colored_frame)

        # Generate the legend
        legend_image = self._generate_patch_legend(patch_colors)

        # Save the legend as a separate image
        cv2.imwrite(str(legend_output_path), legend_image)

    def _generate_patch_legend(self, patch_colors: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Generate a legend image showing the patch indices and their corresponding colors.

        Args:
            patch_colors (Dict[int, np.ndarray]): A dictionary mapping patch indices to their colors.

        Returns:
            np.ndarray: An image of the legend.
        """
        # Parameters for legend layout
        patches_per_column = 11  # Number of patches before starting a new column
        column_width = 150
        row_height = 30

        # Calculate the number of columns and legend dimensions
        num_columns = 2
        # num_columns = (len(patch_colors) + patches_per_column - 1) // patches_per_column
        legend_width = column_width * num_columns
        # legend_width = column_width * num_columns
        legend_height = patches_per_column * row_height + 20
        # legend_height = min(len(patch_colors), patches_per_column) * row_height + 20

        # Create a blank image for the legend
        legend_image = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # White background

        # Font and text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Iterate over the patches and draw the legend
        for idx, (patch_index, color) in enumerate(patch_colors.items()):
            # Determine the row and column position
            col = idx // patches_per_column
            row = idx % patches_per_column

            # Calculate rectangle position for each patch
            start_x = col * column_width + 10
            start_y = 20 + row * row_height
            end_y = start_y + 20

            # Convert RGB to BGR for OpenCV visualization
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))  # Ensure correct BGR order

            # Draw a colored rectangle for each patch
            legend_image[start_y:end_y, start_x:start_x + 40] = bgr_color  # Rectangle for color display

            # Put the text with the patch index next to the color box
            text_position = (start_x + 50, start_y + 15)
            cv2.putText(legend_image, f'Patch {patch_index}', text_position, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return legend_image