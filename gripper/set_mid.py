from vassar_feetech_servo_sdk import ServoController

# Initialize controller with your servo configuration
servo_ids = [1, 2]
controller = ServoController(servo_ids=servo_ids, servo_type="sts")  # or "hls"
controller.connect()

# Read all configured servos
positions = controller.read_all_positions()
for motor_id, pos in positions.items():
    print(f"Motor {motor_id}: {pos} ({pos/4095*100:.1f}%)")

# Set servos to middle position
success = controller.set_middle_position()
if success:
    print("All servos calibrated to middle position!")

controller.disconnect()

# Using context manager
with ServoController([1, 2], "sts") as controller:
    positions = controller.read_all_positions()
    print(positions)
