from vassar_feetech_servo_sdk import find_servo_port

try:
    ports = find_servo_port(return_all=True)
    print(f"Available ports: {ports}")
except Exception as e:
    print(f"No ports found: {e}")
