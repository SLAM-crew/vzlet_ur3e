#!/usr/bin/env python3
"""
sudo chmod 666 /dev/ttyUSB0
create a middle position before that !
"""

from vassar_feetech_servo_sdk import ServoController
import time

def main():
    # Define your servo configuration
    servo_ids = [1, 2]  # Adjust to match your servo IDs
    servo_type = "sts"     # Change to "sts" for STS servos
    
    controller = ServoController(servo_ids=servo_ids, servo_type=servo_type)

    try:
        print(f"Connecting to {servo_type.upper()} servos...")
        controller.connect()
        print("Connected!")
        
        # Read current positions
        print("\n--- Current positions ---")
        positions = controller.read_all_positions()
        for motor_id, pos in sorted(positions.items()):
            print(f"Motor {motor_id}: {pos} ({pos/4095*360:.1f}°)")
        
        # Example 4: Sequential movement
        print("\n--- Sequential movement example ---")
        print("Moving motors one by one...")
        
        positions_sequence = [
            {1: 1784, 2: 2260},           # All to ~middle
            {1: 1360, 2: 2626},        # Motor 1 to middle
        ]
        
        for i, positions in enumerate(positions_sequence):
            print(f"\nStep {i+1}:")
            for motor_id, pos in sorted(positions.items()):
                print(f"  Motor {motor_id}: {pos} ({pos/4095*360:.1f}°)")
            
            controller.write_position(positions)
            time.sleep(1)
        
        # Example 5: Return to middle
        print("\n--- Returning to middle position ---")
        middle_positions = {motor_id: 2048 for motor_id in servo_ids}
        controller.write_position(middle_positions)
        print("All motors returned to middle position")
        
    except ValueError as e:
        print(f"\nError: {e}")
        if "torque limit" in str(e).lower():
            print("Note: Torque limit is only supported for HLS servos.")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # disconnect() will automatically disable all servos
        if 'controller' in locals():
            controller.disconnect()
            print("Disconnected")


if __name__ == "__main__":
    main()
