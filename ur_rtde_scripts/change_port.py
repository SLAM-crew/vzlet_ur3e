from vassar_feetech_servo_sdk import ServoController

# Connect to a servo with current ID 1
controller = ServoController(servo_ids=[1], servo_type="sts")
controller.connect()

# Change its ID from 1 to 10
success = controller.set_motor_id(
    current_id=1,
    new_id=10,
    confirm=True  # Will ask for user confirmation
)

if success:
    print("ID changed! Power cycle the servo to apply.")
    
controller.disconnect()

# After power cycling, connect with new ID
controller = ServoController(servo_ids=[10], servo_type="sts")
controller.connect()
