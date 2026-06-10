#!/usr/bin/env python3

import csv
import os
import sys
import termios
import tty
import time
import signal

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


ROBOT_IP = "172.20.10.4"
CSV_FILE = "recorded_joints.csv"


running = True


def signal_handler(sig, frame):
    global running
    running = False


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return ch


def create_csv_if_needed(csv_path):
    file_exists = os.path.exists(csv_path)

    if not file_exists:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "id",
                "q0",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
            ])


def get_next_index(csv_path):
    if not os.path.exists(csv_path):
        return 1

    with open(csv_path, mode="r") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        return 1

    last_row = rows[-1]

    try:
        return int(last_row[0]) + 1
    except Exception:
        return 1


def append_joint_state(csv_path, idx, joints):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            idx,
            joints[0],
            joints[1],
            joints[2],
            joints[3],
            joints[4],
            joints[5],
        ])


def main():
    global running

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Connecting to robot: {ROBOT_IP}")

    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)

    create_csv_if_needed(CSV_FILE)

    counter = get_next_index(CSV_FILE)

    print("Entering freedrive mode...")
    rtde_c.freedriveMode()

    print("")
    print("Controls:")
    print("  R -> record current joint state")
    print("  Ctrl+C -> exit")
    print("")

    try:
        while running:
            key = get_key()

            if key.lower() == "r":
                joints = rtde_r.getActualQ()

                append_joint_state(
                    CSV_FILE,
                    counter,
                    joints
                )

                print(f"[{counter}] Recorded:")
                print(
                    f"q = "
                    f"{[round(j, 6) for j in joints]}"
                )

                counter += 1

            time.sleep(0.01)

    finally:
        print("\nExiting freedrive mode...")

        try:
            rtde_c.endFreedriveMode()
        except Exception as e:
            print(f"Failed to exit freedrive cleanly: {e}")

        try:
            rtde_c.disconnect()
        except:
            pass

        try:
            rtde_r.disconnect()
        except:
            pass

        print("Disconnected.")


if __name__ == "__main__":
    main()
