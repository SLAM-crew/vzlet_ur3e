#!/usr/bin/env python3


from vassar_feetech_servo_sdk import ServoController
import time


# Define your servo configuration
servo_ids = [1, 2]  # Adjust to match your servo IDs
servo_type = "sts"     # Change to "sts" for STS servos

controller = ServoController(servo_ids=servo_ids, servo_type=servo_type)

print(f"Connecting to {servo_type.upper()} servos...")
controller.connect()
print("Connected!")

# Read current positions
print("\n--- Current positions ---")
positions = controller.read_all_positions()
for motor_id, pos in sorted(positions.items()):
    print(f"Motor {motor_id}: {pos} ({pos/4095*360:.1f}°)")