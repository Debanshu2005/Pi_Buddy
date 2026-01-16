"""
Face Detection Module
Handles face detection using Haar Cascades
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple


class FaceDetector:
    """Handles face detection with multiple strategies"""
    
    def __init__(self, cascade_path: str, config):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better face detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        return gray
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image using multiple strategies"""
        gray = self.preprocess_image(image)
        
        # Multi-scale detection
        faces1 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=self.config.min_face_size,
            maxSize=self.config.max_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces2 = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            maxSize=(500, 500)
        )
        
        # Merge and deduplicate
        all_faces = list(faces1) + list(faces2)
        return self._remove_duplicates(all_faces)
    
    def _remove_duplicates(
        self, 
        faces: List[Tuple[int, int, int, int]], 
        overlap_threshold: int = 30
    ) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping face detections"""
        if len(faces) == 0:
            return []
        
        filtered = []
        for face in faces:
            x, y, w, h = face
            is_duplicate = False
            
            for ex, ey, ew, eh in filtered:
                if (abs(x - ex) < overlap_threshold and 
                    abs(y - ey) < overlap_threshold and
                    abs(w - ew) < overlap_threshold and 
                    abs(h - eh) < overlap_threshold):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(face)
        
        return filtered
    
    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Get the largest detected face"""
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])