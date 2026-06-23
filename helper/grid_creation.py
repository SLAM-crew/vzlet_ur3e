#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# Default global params
INPUT_CSV = "/home/sonieth2/vzlet_ur3e/ws/zone_poses_floor.csv"
OUTPUT_CSV = "grid_poses.csv"
INPUT_POSE_NAME = "mid_storage_00"
# "sensor_storage" / "body_storage", example -->  "body_storage_00" / "body_storage_12"
PREFIX_NAME = "mid_storage"
STORAGE_ID = 4

GRID_X = 3
GRID_Y = 5

# Offset between neighboring cell centers, in meters.

# new version of mid-cell-storage, wide 
OFFSET_X = 0.036
OFFSET_Y = 0.036
# new version of sensor-cell-storage, wide 
# OFFSET_X = 0.033
# OFFSET_Y = 0.033
# new version of body-cell-storage, wide 
# OFFSET_X = 0.042
# OFFSET_Y = 0.042
# old version of body-cell-storage
# OFFSET_X = 0.029
# OFFSET_Y = 0.029

CSV_HEADERS = ["name", "id", "x", "y", "z", "qx", "qy", "qz", "qw"]


def normalize_row(row):
    return {
        (key or "").strip(): (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
        if key is not None
    }


def read_pose_by_name(csv_path, pose_name):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = normalize_row(row)
            if row.get("name") == pose_name:
                return row

    raise ValueError(f"Pose with name '{pose_name}' was not found in {csv_path}")


def pose_value(row, key):
    try:
        return float(row[key])
    except KeyError:
        raise KeyError(f"Missing required CSV column: {key}")
    except ValueError:
        raise ValueError(f"Invalid numeric value for column '{key}': {row.get(key)}")


def make_pose_name(ix, iy):
    if PREFIX_NAME:
        return f"{PREFIX_NAME}_{ix}{iy}"

    return f"{ix}{iy}"


def make_pose_id(storage_id, ix, iy):
    return int(f"{storage_id}{ix}{iy}")


def create_grid_poses(base_pose, grid_x, grid_y, offset_x, offset_y, storage_id):
    base_x = pose_value(base_pose, "x")
    base_y = pose_value(base_pose, "y")
    base_z = pose_value(base_pose, "z")

    qx = pose_value(base_pose, "qx")
    qy = pose_value(base_pose, "qy")
    qz = pose_value(base_pose, "qz")
    qw = pose_value(base_pose, "qw")

    rows = []

    for ix in range(grid_x):
        for iy in range(grid_y):
            rows.append({
                "name": make_pose_name(ix, iy),
                "id": make_pose_id(storage_id, ix, iy),
                "x": base_x + ix * offset_x,
                "y": base_y + iy * offset_y,
                "z": base_z,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
            })

    return rows


def write_poses(csv_path, rows):
    csv_path = Path(csv_path)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def pose_map_from_rows(rows):
    return {
        row["name"]: row
        for row in rows
    }

def plot_grid(rows, grid_x, grid_y):
    poses = pose_map_from_rows(rows)

    fig, ax = plt.subplots()

    # Draw horizontal grid edges.
    for ix in range(grid_x):
        for iy in range(grid_y - 1):
            p1 = poses[make_pose_name(ix, iy)]
            p2 = poses[make_pose_name(ix, iy + 1)]
            ax.plot(
                [p1["x"], p2["x"]],
                [p1["y"], p2["y"]],
                color="black",
                linewidth=1.0,
            )

    # Draw vertical grid edges.
    for ix in range(grid_x - 1):
        for iy in range(grid_y):
            p1 = poses[make_pose_name(ix, iy)]
            p2 = poses[make_pose_name(ix + 1, iy)]
            ax.plot(
                [p1["x"], p2["x"]],
                [p1["y"], p2["y"]],
                color="black",
                linewidth=1.0,
            )

    # Draw vertices.
    xs = [row["x"] for row in rows]
    ys = [row["y"] for row in rows]

    ax.scatter(xs, ys, color="black", zorder=3)

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    grid_width = max(max_x - min_x, abs(OFFSET_X), 0.05)
    grid_height = max(max_y - min_y, abs(OFFSET_Y), 0.05)

    label_offset_x = grid_width * 0.015
    label_offset_y = grid_height * 0.015

    # Draw node labels with XY values.
    for row in rows:
        ax.text(
            row["x"] + label_offset_x,
            row["y"] + label_offset_y,
            f"[{row['name']}]\nid:{row['id']}\n(x:{row['x']:.2f}, y:{row['y']:.2f})",
            fontsize=8,
            va="bottom",
            ha="left",
            zorder=5,
        )

    axis_len = max(grid_width, grid_height) * 0.35

    # Shift axes away from the first grid vertex so they do not overlap the grid.
    axis_origin_x = min_x - abs(OFFSET_X) * 0.75
    axis_origin_y = min_y - abs(OFFSET_Y) * 0.75

    ax.arrow(
        axis_origin_x,
        axis_origin_y,
        axis_len,
        0.0,
        color="red",
        width=0.0005,
        head_width=0.004,
        length_includes_head=True,
        zorder=4,
    )
    ax.arrow(
        axis_origin_x,
        axis_origin_y,
        0.0,
        axis_len,
        color="green",
        width=0.0005,
        head_width=0.004,
        length_includes_head=True,
        zorder=4,
    )

    ax.text(
        axis_origin_x + axis_len,
        axis_origin_y,
        " X",
        color="red",
        va="center",
    )
    ax.text(
        axis_origin_x,
        axis_origin_y + axis_len,
        " Y",
        color="green",
        ha="center",
    )

    pad_x = max(abs(OFFSET_X), grid_width * 0.20)
    pad_y = max(abs(OFFSET_Y), grid_height * 0.20)

    ax.set_xlim(axis_origin_x - pad_x, max_x + pad_x)
    ax.set_ylim(axis_origin_y - pad_y, max_y + pad_y)

    ax.set_title("Generated EEF Pose Grid")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an X*Y EEF pose grid from a single input pose."
    )

    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--pose-name", default=INPUT_POSE_NAME)
    parser.add_argument("--storage-id", type=int, default=STORAGE_ID)
    parser.add_argument("--grid-x", type=int, default=GRID_X)
    parser.add_argument("--grid-y", type=int, default=GRID_Y)
    parser.add_argument("--offset-x", type=float, default=OFFSET_X)
    parser.add_argument("--offset-y", type=float, default=OFFSET_Y)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.grid_x < 1:
        raise ValueError("--grid-x must be >= 1")

    if args.grid_y < 1:
        raise ValueError("--grid-y must be >= 1")

    if args.storage_id < 0:
        raise ValueError("--storage-id must be >= 0")

    base_pose = read_pose_by_name(args.input_csv, args.pose_name)

    rows = create_grid_poses(
        base_pose=base_pose,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        storage_id=args.storage_id,
    )

    write_poses(args.output_csv, rows)

    print(f"Created {len(rows)} grid poses from '{args.pose_name}'")
    print(f"Storage ID: {args.storage_id}")
    print(f"Output CSV: {args.output_csv}")

    plot_grid(
        rows=rows,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
    )


if __name__ == "__main__":
    main()