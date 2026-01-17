"""
Buddy Pi - Hardware Only Service
Face recognition, voice I/O, camera, objects - NO BRAIN
Optimized for Raspberry Pi 4B with Python 3.13
"""
import subprocess
import shutil
import atexit
import cv2
import numpy as np
import time
import logging
import requests
import threading
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Hardware modules only
from config import Config
from states import BuddyState, StateManager
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from stability_tracker import StabilityTracker
from objrecog.obj import ObjectDetector
from servo_controller import ServoController
from motor_controller import MotorController

# Speech imports
import vosk
import json
import edge_tts
import asyncio
import pygame
import io
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play

import os
os.environ["LIBCAMERA_LOG_LEVELS"] = "*:ERROR"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

class BuddyPi:
    """Pi Hardware Service - connects to brain via API"""
    
    def __init__(self, config: Optional[Config] = None, llm_service_url: Optional[str] = None):
        self.config = config or Config.from_env()
        self.llm_service_url = llm_service_url or self.config.llm_service_url
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize camera
        self._init_camera()
        self.csi_process = None
        
        # Initialize face components
        self.detector = FaceDetector(self.config.cascade_path, self.config)
        self.face_recognizer = FaceRecognizer(self.config.model_path, self.config)
        self.stability = StabilityTracker(self.config)
        
        # Initialize object detection
        print(f"🔍 Initializing object detection...")
        self.object_detector = ObjectDetector(confidence_threshold=0.6)
        self.stable_objects = set()
        self.persistent_objects = set()
        print(f"✅ Object detection ready")
        
        # Initialize hardware controllers
        self.servo = ServoController()
        self.motors = MotorController()
        
        # Face tracking state
        self.face_lost_count = 0
        self.max_face_lost = 10
        
        # Initialize speech
        self._init_speech()
        
        # Application state
        self.last_recognition_time = 0
        self.last_object_detection_time = 0
        self.last_object_clear_time = time.time()
        self.current_detections = []
        self.running = False
        self.is_speaking = False
        self.active_user = None
        self.sleep_mode = False
        self.failed_listen_attempts = 0
        self.max_failed_attempts = 4
        
        # Face recognition robustness
        self.recognition_attempts = {}
        self.max_attempts = 5
        self.recognition_threshold = 0.4
        
        print("🤖 Buddy Pi Hardware Ready")
        
        # Initial greeting
        self._startup_greeting()
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format
        )
    
    def _init_camera(self):
        """Initialize camera (CSI or USB) - Pi 4B optimized"""
        self.csi_frame_path = "/tmp/csi_frame.jpg"
        self.use_csi = False
        self.cap = None
        self.csi_process = None

        # Check CSI availability
        if shutil.which("rpicam-hello"):
            print("📷 Starting CSI camera helper...")
            try:
                self.csi_process = subprocess.Popen(
                    ["/usr/bin/python3", "csi_camera_helper.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Wait up to 5 seconds for first frame
                for _ in range(50):
                    if os.path.exists(self.csi_frame_path):
                        self.use_csi = True
                        print("✅ CSI camera ready")
                        atexit.register(self._cleanup_csi)
                        return
                    time.sleep(0.1)
                
                print("⚠️ CSI frame not received, stopping helper")
                self.csi_process.terminate()
                self.csi_process = None
            except Exception as e:
                print(f"⚠️ CSI camera failed: {e}")
                if self.csi_process:
                    self.csi_process.terminate()
                    self.csi_process = None

        # Fallback: USB webcam
        print("📷 Falling back to USB camera")
        self.cap = cv2.VideoCapture(self.config.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open any camera")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera_buffer_size)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        print("✅ USB camera initialized")
    
    def _read_frame(self):
        """Read frame from CSI or USB camera"""
        if self.use_csi and os.path.exists(self.csi_frame_path):
            frame = cv2.imread(self.csi_frame_path)
            return (True, frame) if frame is not None else (False, None)

        if self.cap:
            return self.cap.read()

        return False, None
    
    def _init_speech(self):
        """Initialize speech recognition and TTS - graceful fallback if no audio"""
        try:
            import sounddevice as sd
            
            # Check if any audio devices exist
            try:
                devices = sd.query_devices()
                input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
                
                if not input_devices:
                    print("⚠️ No audio input devices found - speech disabled")
                    self.speech_enabled = False
                    return
                
                print(f"Found {len(input_devices)} input devices")
                
            except Exception as e:
                print(f"⚠️ Audio system unavailable: {e} - speech disabled")
                self.speech_enabled = False
                return
            
            # Try to find working audio configuration
            self.audio_device = None
            self.sample_rate = None
            self.channels = 1  # Start with mono
            
            configs = [
                (44100, 1), (48000, 1), (22050, 1), (16000, 1),
                (44100, 2), (48000, 2), (22050, 2), (16000, 2)
            ]
            
            for device_id in input_devices[:3]:  # Test first 3 devices
                device_info = devices[device_id]
                for rate, channels in configs:
                    if channels > device_info['max_input_channels']:
                        continue
                    
                    try:
                        # Quick test
                        test_audio = sd.rec(
                            int(0.1 * rate),
                            samplerate=rate,
                            channels=channels,
                            dtype="int16",
                            device=device_id
                        )
                        sd.wait()
                        
                        # Found working config
                        self.audio_device = device_id
                        self.sample_rate = rate
                        self.channels = channels
                        print(f"✅ Audio: device {device_id}, {rate}Hz, {channels}ch")
                        break
                        
                    except Exception:
                        continue
                
                if self.audio_device is not None:
                    break
            
            if self.audio_device is None:
                print("⚠️ No working audio configuration - speech disabled")
                self.speech_enabled = False
                return
            
            # Initialize Vosk if audio works
            model_path = "models/vosk-model-small-en-us-0.15"
            if not os.path.exists(model_path):
                print("⚠️ Vosk model not found - speech disabled")
                self.speech_enabled = False
                return
            
            self.vosk_model = vosk.Model(model_path)
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
            
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.tts_voice = "en-IN-NeerjaNeural"
            
            self.speech_enabled = True
            print(f"✅ Speech system ready")
            
        except Exception as e:
            print(f"⚠️ Speech initialization failed: {e} - continuing without speech")
            self.speech_enabled = False
    
    def _detect_sound_direction(self, left_channel, right_channel):
        """Detect sound direction from dual INMP441 microphones"""
        left_energy = np.mean(np.abs(left_channel))
        right_energy = np.mean(np.abs(right_channel))
        
        if left_energy > right_energy * 1.2:
            return "LEFT"
        elif right_energy > left_energy * 1.2:
            return "RIGHT"
        else:
            return "CENTER"
    
    def _startup_greeting(self):
        """Detect and greet person on startup with robust recognition"""
        try:
            print("Starting camera and face recognition...")
            time.sleep(1.0)
            
            recognition_attempts = {}
            
            for attempt in range(50):  # More attempts
                ret, frame = self._read_frame()
                if not ret:
                    continue
                
                faces = self.detector.detect(frame)
                
                if faces:
                    largest_face = self.detector.get_largest_face(faces)
                    is_stable = self.stability.update(largest_face)
                    x, y, w, h = largest_face
                    
                    processed = self._draw_visualization(frame, faces)
                    cv2.imshow('Buddy Vision', processed)
                    cv2.waitKey(1)
                    
                    if w > 80 and h > 80 and is_stable:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.face_recognizer.recognize(face_roi)
                        
                        if name != "Unknown" and confidence > self.recognition_threshold:
                            # Track recognition attempts
                            if name not in recognition_attempts:
                                recognition_attempts[name] = 0
                            recognition_attempts[name] += 1
                            
                            # Confirm recognition after multiple attempts
                            if recognition_attempts[name] >= 3:
                                self.active_user = name
                                print(f"✅ Recognized: {name}")
                                
                                response = self._call_brain_service(
                                    "Hello! Nice to see you!",
                                    recognized_user=name
                                )
                                self._display_response(response)
                                return
                        
                        # Check if we've tried enough times for unknown person
                        total_attempts = sum(recognition_attempts.values())
                        if total_attempts >= self.max_attempts and not self.active_user:
                            print("❓ Unknown person after multiple attempts")
                            response = self._call_brain_service(
                                "I can see someone but don't recognize them. Greet them politely and ask their name.",
                                recognized_user=None
                            )
                            self._display_response(response)
                            return
                
                time.sleep(0.1)
            
            # Fallback greeting
            print("Starting general interaction")
            response = self._call_brain_service("Hello! I'm ready to chat!")
            self._display_response(response)
            
        except Exception as e:
            self.logger.error(f"Startup greeting error: {e}")
            print("Buddy: Hello! I'm ready to chat!")
    
    def _call_brain_service(self, user_input: str, recognized_user: Optional[str] = None) -> dict:
        """Call remote brain service"""
        try:
            # Force face recognition only for identity-related questions (not general knowledge)
            user_lower = user_input.lower()
            identity_patterns = [
                'who am i', 'who is this', 'who is here', 'who do you see',
                'recognize me', 'recognize this', 'can you see me', 'do you know me',
                'identify me', 'identify this', 'scan face', 'scan me'
            ]
            if any(pattern in user_lower for pattern in identity_patterns):
                recognized_user = self._force_face_recognition()
                if recognized_user:
                    print(f"🔍 Forced recognition: {recognized_user}")
            
            # Force fresh object detection if user is asking about objects
            objects = []
            if any(phrase in user_input.lower() for phrase in 
                  ['what is', 'what do you see', 'in my hand', 'holding', 'show you']):
                # Get fresh object detection immediately
                ret, frame = self._read_frame()
                if ret:
                    fresh_detections = self.object_detector.detect(frame)
                    if fresh_detections:
                        objects = [det['name'] for det in fresh_detections]
                        # Update persistent objects too
                        self.persistent_objects.update(objects)
                        print(f"🔄 Fresh detection: {objects}")
            
            # Fallback to cached objects if no fresh detection
            if not objects:
                objects = list(self.persistent_objects) if self.persistent_objects else list(self.stable_objects)
            
            # Prepare request
            request_data = {
                "user_input": user_input,
                "recognized_user": recognized_user,
                "objects_visible": objects
            }
            
            # Call brain API
            response = requests.post(
                f"{self.llm_service_url}/chat",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Brain service error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Brain service call failed: {e}")
            return {
                "reply": "Sorry, I'm having trouble thinking right now.",
                "intent": "conversation",
                "emotion": "apologetic",
                "raw_response": ""
            }
    
    def _force_face_recognition(self) -> Optional[str]:
        """Force immediate face recognition, can handle multiple faces"""
        try:
            ret, frame = self._read_frame()
            if not ret:
                return None
            
            # Add error handling for face detection
            try:
                faces = self.detector.detect(frame)
            except cv2.error as e:
                print(f"Face detection error: {e}")
                return None
            except Exception as e:
                print(f"Face detection error: {e}")
                return None
            
            if not faces:
                return None
            
            recognized_names = []
            
            # Process all faces in frame
            for (x, y, w, h) in faces:
                if w > 80 and h > 80:  # Minimum face size
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.face_recognizer.recognize(face_roi)
                        
                        if name != "Unknown" and confidence > self.recognition_threshold:
                            recognized_names.append(name)
                            print(f"✅ Recognized: {name} (confidence: {confidence:.2f})")
                    except Exception as e:
                        print(f"Face recognition error: {e}")
                        continue
            
            # Update active user to the first recognized person
            if recognized_names:
                self.active_user = recognized_names[0]
                if len(recognized_names) == 1:
                    return recognized_names[0]
                else:
                    # Multiple people detected
                    names_str = ", ".join(recognized_names)
                    print(f"👥 Multiple people: {names_str}")
                    return recognized_names[0]  # Return primary user
            
            return None
            
        except Exception as e:
            print(f"Force recognition error: {e}")
            return None
    
    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool, Optional[str], float]:
        """Process frame for face detection and recognition"""
        try:
            faces = self.detector.detect(frame)
        except (cv2.error, Exception) as e:
            print(f"Face detection error: {e}")
            faces = []
        
        name = None
        confidence = 0.0
        face_detected = len(faces) > 0
        
        if face_detected:
            largest_face = self.detector.get_largest_face(faces)
            is_stable = self.stability.update(largest_face)
            
            current_time = time.time()
            if (is_stable and 
                (current_time - self.last_recognition_time) > 30.0):  # 30 second intervals
                
                x, y, w, h = largest_face
                if w > self.config.min_face_size[0] and h > self.config.min_face_size[1]:
                    face_roi = frame[y:y+h, x:x+w]
                    name, confidence = self.face_recognizer.recognize(face_roi)
                    
                    # Update active user if face is recognized during conversation
                    if name != "Unknown" and confidence > self.recognition_threshold:
                        if self.active_user != name:
                            self.active_user = name
                        # Silent recognition - no console spam
                    
                    self.last_recognition_time = current_time
        else:
            self.stability.reset()
        
        # Process object detection less frequently
        current_time = time.time()
        if (current_time - self.last_object_detection_time) > 5.0:
            self.current_detections = self._process_objects(frame)
            self.last_object_detection_time = current_time
        
        # Clear persistent objects every 60 seconds (longer retention)
        if (current_time - self.last_object_clear_time) > 60.0:
            print(f"🧹 Clearing old objects: {self.persistent_objects}")
            self.persistent_objects.clear()
            self.last_object_clear_time = current_time
        
        # Draw visualization with cached objects
        processed = self._draw_visualization(frame, faces, name, confidence, self.current_detections)
        
        return processed, face_detected, name, confidence
    
    def _process_objects(self, frame):
        """Process object detection with debug output"""
        try:
            detections = self.object_detector.detect(frame)
            if detections:
                current_objects = set([det['name'] for det in detections])
                confidence_info = [(det['name'], det['confidence']) for det in detections]
                print(f"🔍 Objects detected: {confidence_info}")
                
                self.stable_objects = current_objects
                self.persistent_objects.update(current_objects)
                return detections
            else:
                # Only clear stable objects if no detection, keep persistent ones
                if self.stable_objects:
                    print(f"🚫 No objects detected this frame")
                self.stable_objects = set()
                return []
        except Exception as e:
            print(f"⚠️ Object detection error: {e}")
            self.stable_objects = set()
            return []
    
    def speak(self, text):
        """Convert text to speech using Edge TTS"""
        if not self.speech_enabled or not text:
            return
        
        def _speak():
            try:
                self.is_speaking = True
                asyncio.run(self._generate_and_play_speech(text))
                self.is_speaking = False
            except Exception as e:
                print(f"TTS Error: {e}")
                self.is_speaking = False
        
        threading.Thread(target=_speak, daemon=True).start()
    
    async def _generate_and_play_speech(self, text):
        """Generate speech using Edge TTS and play via Pi hardware"""
        try:
            communicate = edge_tts.Communicate(text, self.tts_voice)
            
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if audio_data:
                # Save to temp file and play via Pi hardware with specific device
                temp_audio = "/tmp/tts_output.wav"
                with open(temp_audio, "wb") as f:
                    f.write(audio_data)
                
                # Play using aplay with specific audio device
                os.system(f"aplay -D hw:0,0 {temp_audio}")
                    
        except Exception as e:
            print(f"Edge TTS Error: {e}")
    
    def listen_for_speech(self, timeout=6.0):
        """Listen for speech input using dual INMP441 microphones with Vosk"""
        if not self.speech_enabled:
            return ""
        
        try:
            print("🎤 Listening with dual INMP441...")
            
            # Record audio using dual INMP441 microphones
            audio = sd.rec(
                int(timeout * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.audio_device
            )
            sd.wait()
            
            # Separate channels for direction detection
            left_channel = audio[:, 0]
            right_channel = audio[:, 1]
            
            # Detect sound direction
            direction = self._detect_sound_direction(left_channel, right_channel)
            
            # Mix both channels for recognition (boost volume)
            mixed_audio = ((left_channel.astype(np.float32) + right_channel.astype(np.float32)) / 2)
            mixed_audio = np.clip(mixed_audio * 2.0, -32768, 32767).astype(np.int16)  # Boost volume
            
            print("Processing speech with Vosk...")
            
            # Process entire audio buffer with Vosk
            self.vosk_rec.AcceptWaveform(mixed_audio.tobytes())
            result = json.loads(self.vosk_rec.FinalResult())
            text = result.get('text', '').strip()
            
            if text:
                print(f"You said ({direction}): '{text}'")
                return text
            
            print("Couldn't understand that.")
            return ""
            
        except Exception as e:
            print(f"INMP441 audio error: {e}")
            return ""
    
    def _draw_visualization(self, frame, faces, name=None, confidence=0.0, detections=None):
        """Draw UI overlays with object detection boxes"""
        # Draw object detection boxes FIRST (so they appear behind face boxes)
        if detections:
            frame = self.object_detector.draw_detections(frame, detections)
        
        # Draw face detection
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if self.stability.is_stable else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            if name and name != "Unknown":
                label = f"{name} ({confidence:.0%})"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status display
        if self.active_user:
            status = f"💬 Chatting with: {self.active_user}"
            color = (0, 255, 0)
        else:
            status = "🔍 Looking for people"
            color = (0, 255, 255)
        
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show detected objects in text with more info
        if self.persistent_objects:
            objects_text = f"Objects: {', '.join(list(self.persistent_objects)[:3])}"
            cv2.putText(frame, objects_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Show current frame objects
        if self.stable_objects:
            current_text = f"Current: {', '.join(list(self.stable_objects)[:2])}"
            cv2.putText(frame, current_text, (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame
    
    def _display_response(self, response: dict):
        """Display AI response and handle TTS"""
        if not response:
            return
        
        reply = response.get("reply", "")
        intent = response.get("intent", "conversation")
        raw_response = response.get("raw_response", "")
        
        # Show the full JSON response like original
        print(f"\nBuddy: {raw_response if raw_response else reply}")
        
        if intent != "conversation":
            print(f"[INTENT: {intent}]")
            
            # Handle movement commands
            if intent in ["move_forward", "move_backward", "move_left", "move_right", "stop"]:
                self._execute_movement(intent)
        
        # Start TTS with just the reply text
        self.speak(reply)
        
        # Dynamic wait time based on reply length
        words = len(reply.split())
        chars = len(reply)
        
        # Calculate reading time: ~200 words per minute + TTS time
        reading_time = max(2.0, (words * 0.3) + (chars * 0.05))  # More accurate timing
        reading_time = min(reading_time, 8.0)  # Cap at 8 seconds max
        
        threading.Timer(reading_time, self._delayed_listening).start()
    
    def _execute_movement(self, intent: str):
        """Execute movement commands based on intent"""
        movement_map = {
            "move_forward": lambda: self.motors.move_forward(1.0),
            "move_backward": lambda: self.motors.move_backward(1.0),
            "move_left": lambda: self.motors.turn_left(0.5),
            "move_right": lambda: self.motors.turn_right(0.5),
            "stop": lambda: self.motors.stop()
        }
        
        if intent in movement_map:
            print(f"🤖 Executing: {intent}")
            movement_map[intent]()
    
    def _play_thinking_sound(self):
        """Play thinking sound while processing"""
        thinking_sounds = ["Hmm", "Let me think", "Umm", "Well"]
        import random
        sound = random.choice(thinking_sounds)
        self.speak(sound)
        time.sleep(0.3)  # Shorter wait
    
    def _delayed_listening(self):
        """Start listening after a delay"""
        if not self.running or self.sleep_mode:
            return
            
        user_text = self.listen_for_speech(timeout=6.0)
        
        if user_text:
            self.failed_listen_attempts = 0  # Reset counter on successful input
            self._play_thinking_sound()
            threading.Thread(target=self._process_input, args=(user_text,), daemon=True).start()
        else:
            self.failed_listen_attempts += 1
            print(f"🔇 No input attempt {self.failed_listen_attempts}/{self.max_failed_attempts}")
            
            if self.failed_listen_attempts >= self.max_failed_attempts:
                self._enter_sleep_mode()
            elif self.running:
                threading.Timer(1.0, self._delayed_listening).start()
    
    def _process_input(self, user_text):
        """Process user input"""
        try:
            # Check if user is introducing themselves (be more specific)
            if any(phrase in user_text.lower() for phrase in ['my name is', 'i am', "i'm"]):
                # Extract name and register face - only single words
                name = self._extract_name(user_text)
                if name and len(name.split()) == 1 and name.isalpha() and len(name) > 1:
                    if self.active_user != name:
                        # Use the last processed frame instead of capturing new one
                        if hasattr(self, 'last_frame') and self.last_frame is not None:
                            faces = self.detector.detect(self.last_frame)
                            if faces:
                                largest_face = self.detector.get_largest_face(faces)
                                x, y, w, h = largest_face
                                if w > 80 and h > 80:
                                    face_roi = self.last_frame[y:y+h, x:x+w]
                                    if self.face_recognizer.add_face(name, face_roi):
                                        self.active_user = name
                                        print(f"Registered new face for: {name}")
            
            response = self._call_brain_service(user_text, self.active_user)
            self._display_response(response)
        except Exception as e:
            print(f"Error processing input: {e}")
    
    def _enter_sleep_mode(self):
        """Trigger sleep mode - ends conversation loop"""
        print("\n😴 Entering sleep mode...")
        self.sleep_mode = True
        # The conversation loop will detect this and exit, starting sleep loop
    
    def _wake_up_and_restart(self):
        """Wake up and restart conversation loop"""
        print("😊 Waking up!")
        self.sleep_mode = False
        self.failed_listen_attempts = 0
        
        # Quick greeting
        self.speak("I'm awake! Let me see who's here.")
        time.sleep(1)
        
        # Quick face recognition
        recognized_user = self._force_face_recognition()
        if recognized_user:
            self.active_user = recognized_user
            response = self._call_brain_service(
                "I just woke up. Greet them warmly.",
                recognized_user=recognized_user
            )
        else:
            response = self._call_brain_service("I just woke up. Say hello.")
        
        self._display_response(response)
        # The sleep loop will detect sleep_mode=False and exit, restarting conversation loop
    
    def _extract_name(self, text: str) -> str:
        """Extract name from introduction text"""
        text = text.lower()
        if 'my name is' in text:
            return text.split('my name is')[1].strip().title()
        elif 'i am' in text:
            return text.split('i am')[1].strip().title()
        elif "i'm" in text:
            return text.split("i'm")[1].strip().title()
        return ""
    
    def _cleanup_csi(self):
        """Cleanup CSI camera process"""
        if hasattr(self, 'csi_process') and self.csi_process:
            self.csi_process.terminate()
    
    def run(self):
        """Main application loop with independent sleep/wake cycles"""
        self.running = True
        
        try:
            # Start with conversation loop
            self._conversation_loop()
        except KeyboardInterrupt:
            print("\n⚠️ Keyboard interrupt received")
            self.running = False
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            self.cleanup()
    
    def _conversation_loop(self):
        """Active conversation loop with camera and face detection"""
        print("🎯 Starting conversation loop")
        self.last_frame = None
        self.failed_listen_attempts = 0
        
        while self.running and not self.sleep_mode:
            # Camera and face detection
            ret, frame = self._read_frame()
            if not ret:
                continue
            
            self.last_frame = frame.copy()
            processed, face_detected, name, confidence = self._process_frame(frame)
            cv2.imshow('Buddy Vision', processed)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
        
        # If we exit due to sleep mode, start sleep loop
        if self.sleep_mode and self.running:
            self._sleep_loop()
    
    def _sleep_loop(self):
        """Independent sleep loop - only wake word detection"""
        print("😴 Starting sleep loop")
        
        # Close OpenCV window
        try:
            cv2.destroyAllWindows()
            print("OpenCV window closed")
        except:
            pass
        
        # Say goodnight
        self.speak("Going to sleep. Say 'Hey Buddy' to wake me up.")
        time.sleep(2)
        
        print("😴 Sleep mode active. Listening for 'Hey Buddy'...")
        
        # Sleep loop - only wake word detection
        while self.running and self.sleep_mode:
            try:
                print("[SLEEP] Listening for wake word...")
                
                # Record audio using INMP441 microphones
                audio = sd.rec(
                    int(2.0 * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    device=self.audio_device
                )
                sd.wait()
                
                # Mix channels for Vosk processing
                left_channel = audio[:, 0].astype(np.float32)
                right_channel = audio[:, 1].astype(np.float32)
                mixed_audio = ((left_channel + right_channel) / 2).astype(np.int16)
                
                print("[SLEEP] Processing audio...")
                try:
                    # Boost volume and process with Vosk
                    boosted_audio = np.clip(mixed_audio.astype(np.float32) * 2.0, -32768, 32767).astype(np.int16)
                    
                    # Process entire buffer and get final result
                    self.vosk_rec.AcceptWaveform(boosted_audio.tobytes())
                    result = json.loads(self.vosk_rec.FinalResult())
                    text = result.get('text', '').strip()
                    
                    if text:
                        text_lower = text.lower()
                        print(f"[SLEEP] Heard: '{text}'")
                        
                        # Check for wake words
                        if any(phrase in text_lower for phrase in ['hey buddy', 'wake up', 'buddy wake up']):
                            print(f"✅ [SLEEP] Wake word detected: '{text}'")
                            self._wake_up_and_restart()
                            break
                        else:
                            print(f"❌ [SLEEP] Not a wake word: '{text}'")
                    else:
                        print("🔇 [SLEEP] Could not understand audio")
                        
                except Exception as e:
                    print(f"❌ [SLEEP] Vosk error: {e}")
                    time.sleep(0.5)
                    
            except KeyboardInterrupt:
                print("\n⚠️ [SLEEP] Interrupted during sleep mode")
                self.running = False
                break
            except Exception as e:
                print(f"❌ [SLEEP] Wake detection error: {e}")
                time.sleep(0.5)
        
        # If we exit sleep due to wake up, restart conversation loop
        if not self.sleep_mode and self.running:
            self._conversation_loop()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        try:
            requests.post(f"{self.llm_service_url}/clear", timeout=5)
        except:
            pass
        
        if self.use_csi and hasattr(self, 'csi_process') and self.csi_process:
            self.csi_process.terminate()
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        try:
            cv2.destroyAllWindows()
            pygame.mixer.quit()
        except:
            pass
            
        print("\nBuddy Pi: Goodbye!")

def main():
    """Entry point"""
    try:
        config = Config.from_env()
        
        llm_url = config.llm_service_url
        
        print(f"🤖 Starting Buddy Pi...")
        print(f"🔗 LLM Service: {llm_url}")
        print(f"📷 Camera: {config.camera_index}")
        
        buddy = BuddyPi(config)
        buddy.run()
        
        return 0
    
    except Exception as e:
        logging.error(f"Startup failed: {e}")
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
