import RPi.GPIO as GPIO
import time

class MotorController:
    def __init__(self, left_pins=(18, 19), right_pins=(20, 21)):
        self.left_forward, self.left_backward = left_pins
        self.right_forward, self.right_backward = right_pins
        
        GPIO.setmode(GPIO.BCM)
        
        # Setup motor pins
        for pin in [self.left_forward, self.left_backward, self.right_forward, self.right_backward]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        print("🚗 Motor controller initialized")
    
    def move_forward(self, duration=1.0):
        """Move robot forward"""
        print("⬆️ Moving forward")
        GPIO.output(self.left_forward, GPIO.HIGH)
        GPIO.output(self.right_forward, GPIO.HIGH)
        time.sleep(duration)
        self.stop()
    
    def move_backward(self, duration=1.0):
        """Move robot backward"""
        print("⬇️ Moving backward")
        GPIO.output(self.left_backward, GPIO.HIGH)
        GPIO.output(self.right_backward, GPIO.HIGH)
        time.sleep(duration)
        self.stop()
    
    def turn_left(self, duration=0.5):
        """Turn robot left"""
        print("⬅️ Turning left")
        GPIO.output(self.right_forward, GPIO.HIGH)
        GPIO.output(self.left_backward, GPIO.HIGH)
        time.sleep(duration)
        self.stop()
    
    def turn_right(self, duration=0.5):
        """Turn robot right"""
        print("➡️ Turning right")
        GPIO.output(self.left_forward, GPIO.HIGH)
        GPIO.output(self.right_backward, GPIO.HIGH)
        time.sleep(duration)
        self.stop()
    
    def stop(self):
        """Stop all motors"""
        GPIO.output(self.left_forward, GPIO.LOW)
        GPIO.output(self.left_backward, GPIO.LOW)
        GPIO.output(self.right_forward, GPIO.LOW)
        GPIO.output(self.right_backward, GPIO.LOW)
    
    def cleanup(self):
        """Cleanup GPIO"""
        self.stop()
        GPIO.cleanup()
        print("🚗 Motor cleanup complete")