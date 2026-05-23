import rtde_receive

BASE_IP = "192.168.31.59"  
rtde_r = rtde_receive.RTDEReceiveInterface(BASE_IP)
actual_q = rtde_r.getActualQ()
print(f"Actual joint positions: {actual_q}")
