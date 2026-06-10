#!/usr/bin/env python3

import csv
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

from scipy.spatial.transform import Rotation as R

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


ROBOT_IP = "172.20.10.4"
CSV_FILE = "recorded_poses.csv"


rtde_c = RTDEControlInterface(ROBOT_IP)
rtde_r = RTDEReceiveInterface(ROBOT_IP)


original_terminal_settings = termios.tcgetattr(sys.stdin)


def restore_terminal():
    termios.tcsetattr(
        sys.stdin,
        termios.TCSADRAIN,
        original_terminal_settings
    )


def cleanup(sig=None, frame=None):

    print("\nLeaving teach mode...")

    restore_terminal()

    try:
        rtde_c.endTeachMode()
    except Exception as e:
        print(f"endTeachMode failed: {e}")

    try:
        rtde_c.stopScript()
    except Exception:
        pass

    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)


def ensure_csv_exists():

    if Path(CSV_FILE).exists():
        return

    with open(CSV_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "id",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw"
        ])


def get_next_counter():

    if not Path(CSV_FILE).exists():
        return 1

    with open(CSV_FILE, "r") as f:

        rows = list(csv.reader(f))

    if len(rows) <= 1:
        return 1

    try:
        return int(rows[-1][0]) + 1
    except Exception:
        return 1


def rotvec_to_quaternion(rx, ry, rz):

    rotation = R.from_rotvec([rx, ry, rz])

    qx, qy, qz, qw = rotation.as_quat()

    return qx, qy, qz, qw


def get_key_nonblocking(timeout=0.05):

    dr, _, _ = select.select([sys.stdin], [], [], timeout)

    if dr:
        return sys.stdin.read(1)

    return None


def main():

    ensure_csv_exists()

    counter = get_next_counter()

    print("Entering teach mode...")

    success = rtde_c.teachMode()

    if not success:
        print("Failed to enter teach mode")
        return

    tty.setcbreak(sys.stdin.fileno())

    print("Robot is now in teach/freedrive mode")
    print("Press 'r' to record current TCP pose")
    print("Press CTRL+C to exit")

    while True:

        key = get_key_nonblocking()

        if key is None:
            continue

        if key.lower() == "r":

            tcp_pose = rtde_r.getActualTCPPose()

            x = tcp_pose[0]
            y = tcp_pose[1]
            z = tcp_pose[2]

            rx = tcp_pose[3]
            ry = tcp_pose[4]
            rz = tcp_pose[5]

            qx, qy, qz, qw = rotvec_to_quaternion(
                rx,
                ry,
                rz
            )

            row = [
                counter,
                x,
                y,
                z,
                qx,
                qy,
                qz,
                qw
            ]

            with open(CSV_FILE, "a", newline="") as f:

                writer = csv.writer(f)
                writer.writerow(row)

            print(f"Recorded pose #{counter}")
            print(row)

            counter += 1


if __name__ == "__main__":
    main()
