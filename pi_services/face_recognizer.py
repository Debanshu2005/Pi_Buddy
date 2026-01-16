"""
Face Recognition Module
Handles face recognition using ONNX model
"""

import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
import logging
import html
from typing import Dict, Tuple
from pi_memory import save_face, get_all_faces


class FaceRecognizer:
    """Handles face recognition using ONNX model"""
    
    def __init__(self, model_path: str, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load ONNX model
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, session_options)
        self.input_name = self.session.get_inputs()[0].name
        
        self.known_faces: Dict[str, np.ndarray] = {}
        self.load_known_faces()
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for ONNX model"""
        face_resized = cv2.resize(face_img, (112, 112))
        face_normalized = (face_resized - 127.5) / 128.0
        return np.expand_dims(
            face_normalized.transpose(2, 0, 1), 
            axis=0
        ).astype(np.float32)
    
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract face embedding from image"""
        preprocessed = self.preprocess_face(face_img)
        embedding = self.session.run(None, {self.input_name: preprocessed})[0]
        return embedding.flatten()
    
    def recognize(self, face_img: np.ndarray) -> Tuple[str, float]:
        """Recognize a face from image"""
        try:
            # Enhance image
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_enhanced = cv2.equalizeHist(face_gray)
            face_enhanced = cv2.cvtColor(face_enhanced, cv2.COLOR_GRAY2BGR)
            
            embedding = self.get_embedding(face_enhanced)
            
            if not self.known_faces:
                return "Unknown", 0.0
            
            best_match = None
            min_distance = float('inf')
            
            for name, known_embedding in self.known_faces.items():
                distance = cosine(embedding, known_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = name
            
            confidence = 1 - min_distance
            
            if min_distance < self.config.recognition_threshold:
                return best_match, confidence
            
            return "Unknown", 0.0
            
        except Exception as e:
            self.logger.error(f"Recognition error: {e}", exc_info=True)
            return "Unknown", 0.0
    
    def add_face(self, name: str, face_img: np.ndarray) -> bool:
        """Add a new face to known faces"""
        try:
            print(f"DEBUG: add_face called for '{name}' with image shape {face_img.shape}")
            embedding = self.get_embedding(face_img)
            print(f"DEBUG: Generated embedding shape: {embedding.shape}")
            
            self.known_faces[name] = embedding
            print(f"DEBUG: Added to known_faces dict, total faces: {len(self.known_faces)}")
            
            save_face(name, embedding, 0.95)
            print(f"DEBUG: Called save_face for '{name}'")
            
            self.logger.info(f"Added face for user: {name}")
            return True
        except Exception as e:
            print(f"DEBUG: Exception in add_face: {e}")
            self.logger.error(f"Failed to add face for {name}: {e}", exc_info=True)
            return False
    
    def load_known_faces(self) -> None:
        """Load known faces from database"""
        try:
            faces = get_all_faces()
            self.known_faces = {
                html.unescape(name): embedding 
                for name, embedding in faces.items()
            }
            print(f"INFO:face_recognizer:Loaded {len(self.known_faces)} known faces")
            if self.known_faces:
                print(f"Known faces: {list(self.known_faces.keys())}")
            else:
                print("WARNING: No faces found in database!")
            self.logger.info(f"Loaded {len(self.known_faces)} known faces")
        except Exception as e:
            print(f"ERROR loading faces: {e}")
            self.logger.error(f"Failed to load known faces: {e}", exc_info=True)
            self.known_faces = {}
