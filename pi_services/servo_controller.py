import RPi.GPIO as GPIO
import time
import threading

class ServoController:
    def __init__(self, servo_pin=17):
        self.servo_pin = servo_pin
        self.current_angle = 90  # Start at center
        self.target_angle = 90
        self.scanning = False
        self.scan_direction = 1
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servo_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.servo_pin, 50)
        self.pwm.start(0)
        
        # Move to center position
        self.set_angle(90)
        print("🔧 Servo initialized at center position")
    
    def set_angle(self, angle):
        """Set servo to specific angle (0-180)"""
        angle = max(0, min(180, angle))
        duty = 2 + (angle / 18)
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.3)
        self.pwm.ChangeDutyCycle(0)
        self.current_angle = angle
    
    def track_face(self, face_x, frame_width):
        """Track face by adjusting servo angle"""
        if not face_x or not frame_width:
            return
        
        # Calculate face position relative to center
        center_x = frame_width // 2
        offset = face_x - center_x
        
        # Convert offset to angle adjustment
        angle_adjustment = (offset / center_x) * 30  # Max 30 degree adjustment
        
        # Calculate new target angle
        new_angle = self.current_angle - angle_adjustment
        new_angle = max(30, min(150, new_angle))  # Limit range
        
        # Only move if significant change
        if abs(new_angle - self.current_angle) > 5:
            print(f"📹 Tracking face: {new_angle:.0f}°")
            self.set_angle(new_angle)
    
    def start_scanning(self):
        """Start scanning for faces"""
        if not self.scanning:
            self.scanning = True
            threading.Thread(target=self._scan_loop, daemon=True).start()
            print("🔍 Started face scanning")
    
    def stop_scanning(self):
        """Stop scanning"""
        self.scanning = False
        print("⏹️ Stopped face scanning")
    
    def _scan_loop(self):
        """Continuous scanning loop"""
        while self.scanning:
            if self.current_angle >= 150:
                self.scan_direction = -1
            elif self.current_angle <= 30:
                self.scan_direction = 1
            
            new_angle = self.current_angle + (self.scan_direction * 15)
            self.set_angle(new_angle)
            time.sleep(1)
    
    def cleanup(self):
        """Cleanup GPIO"""
        self.scanning = False
        self.pwm.stop()
        GPIO.cleanup()
        print("🔧 Servo cleanup complete")