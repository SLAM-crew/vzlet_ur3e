import rtde_control
import rtde_receive
import time

BASE_IP = "192.168.31.59" 
rtde_c = rtde_control.RTDEControlInterface(BASE_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(BASE_IP)

try:
    current_q = rtde_r.getActualQ()
    print(f"Current joint positions: {current_q}")
    home_q = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
    rtde_c.moveJ(home_q, speed=0.3, acceleration=0.3)
    print("Reached home position")
    
    joint_q = [-1.54, -1.83, -2.28, -0.59, 1.60, 0.023]
    print("Moving to servo starting position...")
    rtde_c.moveJ(joint_q, speed=1.0, acceleration=0.5) # moveJ = move joints
    print("Ready to start servo control")
    
    # SERVO CONTROL PARAMETERS
    velocity = 0.2
    acceleration = 0.5
    dt = 1.0/500  # ~ 2ms
    lookahead_time = 0.1
    gain = 300
    
    # 6. EXECUTE SERVO CONTROL LOOP
    print("Starting servo control loop...")
    for i in range(1000):
        t_start = rtde_c.initPeriod()
        rtde_c.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain)
        joint_q[0] += 0.001
        joint_q[1] += 0.001
        rtde_c.waitPeriod(t_start)
        
        # Optional: Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}/1000")
    
    # 7. STOP SERVO AND RETURN TO HOME (OPTIONAL)
    print("Stopping servo control...")
    rtde_c.servoStop()
    
    # return back to home
    # rtde_c.moveJ(home_q, speed=1.0, acceleration=0.5)
    rtde_c.stopScript()

except KeyboardInterrupt:
    print("\nInterrupted by user")
    rtde_c.servoStop()
    rtde_c.stopScript()
    
except Exception as e:
    print(f"Error occurred: {e}")
    try:
        rtde_c.servoStop()
        rtde_c.stopScript()
    except:
        pass
