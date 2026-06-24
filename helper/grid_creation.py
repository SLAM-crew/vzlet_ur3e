#!/usr/bin/env python3

import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Type of grid: 2d / 1d
GRID_TYPE = 1

INPUT_CSV = "/home/sonieth2/vzlet_ur3e/ws/zone_poses_floor.csv"
OUTPUT_CSV = "grid_poses.csv"
INPUT_POSE_NAME = "wire5_storage_00"
# "sensor_storage" / "body_storage", example -->  "body_storage_00" / "body_storage_12"
PREFIX_NAME = "wire5_storage"
STORAGE_ID = 5

GRID_X = 5
GRID_Y = 0

# Offset between neighboring cell centers, in meters.
OFFSET_X = 0.019
OFFSET_Y = 0.0      # if in case of 1d --> it wont use this axis
# mid-cell-storage
# OFFSET_X = 0.036
# OFFSET_Y = 0.036
# new version of sensor-cell-storage
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


def make_pose_name(ix, iy=None, grid_type=GRID_TYPE):
    if grid_type == 1:
        if PREFIX_NAME:
            return f"{PREFIX_NAME}_{ix:02d}"

        return f"{ix:02d}"

    if PREFIX_NAME:
        return f"{PREFIX_NAME}_{ix}{iy}"

    return f"{ix}{iy}"


def make_pose_id(storage_id, ix, iy=None, grid_type=GRID_TYPE):
    if grid_type == 1:
        return int(f"{storage_id}{ix:02d}")

    return int(f"{storage_id}{ix}{iy}")


def create_grid_poses(base_pose, grid_x, grid_y, offset_x, offset_y, storage_id, grid_type):
    base_x = pose_value(base_pose, "x")
    base_y = pose_value(base_pose, "y")
    base_z = pose_value(base_pose, "z")

    qx = pose_value(base_pose, "qx")
    qy = pose_value(base_pose, "qy")
    qz = pose_value(base_pose, "qz")
    qw = pose_value(base_pose, "qw")

    rows = []

    if grid_type == 1:
        x_is_zero = abs(offset_x) < 1e-12
        y_is_zero = abs(offset_y) < 1e-12

        if x_is_zero and y_is_zero:
            raise ValueError("For 1d grid, one offset must be non-zero")

        if not x_is_zero and not y_is_zero:
            raise ValueError("For 1d grid, one offset must be zero")

        count = grid_y if x_is_zero else grid_x

        for index in range(count):
            rows.append({
                "name": make_pose_name(index, grid_type=grid_type),
                "id": make_pose_id(storage_id, index, grid_type=grid_type),
                "x": base_x + index * offset_x,
                "y": base_y + index * offset_y,
                "z": base_z,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
            })

        return rows

    for ix in range(grid_x):
        for iy in range(grid_y):
            rows.append({
                "name": make_pose_name(ix, iy, grid_type),
                "id": make_pose_id(storage_id, ix, iy, grid_type),
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
        # writer.writeheader() # no need of csv headers
        writer.writerows(rows)


def pose_map_from_rows(rows):
    return {
        row["name"]: row
        for row in rows
    }

def plot_grid(rows, grid_x, grid_y, grid_type, offset_x, offset_y):
    poses = pose_map_from_rows(rows)

    fig, ax = plt.subplots()

    if grid_type == 1:
        for index in range(len(rows) - 1):
            p1 = rows[index]
            p2 = rows[index + 1]
            ax.plot(
                [p1["x"], p2["x"]],
                [p1["y"], p2["y"]],
                color="black",
                linewidth=1.0,
            )
    else:
        # Draw horizontal grid edges.
        for ix in range(grid_x):
            for iy in range(grid_y - 1):
                p1 = poses[make_pose_name(ix, iy, grid_type)]
                p2 = poses[make_pose_name(ix, iy + 1, grid_type)]
                ax.plot(
                    [p1["x"], p2["x"]],
                    [p1["y"], p2["y"]],
                    color="black",
                    linewidth=1.0,
                )

        # Draw vertical grid edges.
        for ix in range(grid_x - 1):
            for iy in range(grid_y):
                p1 = poses[make_pose_name(ix, iy, grid_type)]
                p2 = poses[make_pose_name(ix + 1, iy, grid_type)]
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

    grid_width = max(max_x - min_x, abs(offset_x), 0.05)
    grid_height = max(max_y - min_y, abs(offset_y), 0.05)

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
    axis_origin_x = min_x - max(abs(offset_x), grid_width * 0.20) * 0.75
    axis_origin_y = min_y - max(abs(offset_y), grid_height * 0.20) * 0.75

    axis_x_end = axis_origin_x + axis_len
    axis_y_end = axis_origin_y + axis_len

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
        axis_x_end,
        axis_origin_y,
        " X",
        color="red",
        va="center",
    )
    ax.text(
        axis_origin_x,
        axis_y_end,
        " Y",
        color="green",
        ha="center",
    )

    pad_x = max(abs(offset_x), grid_width * 0.20, axis_len * 0.35)
    pad_y = max(abs(offset_y), grid_height * 0.20, axis_len * 0.35)

    plot_min_x = min(min_x, axis_origin_x)
    plot_max_x = max(max_x, axis_x_end)
    plot_min_y = min(min_y, axis_origin_y)
    plot_max_y = max(max_y, axis_y_end)

    ax.set_xlim(plot_min_x - pad_x, plot_max_x + pad_x)
    ax.set_ylim(plot_min_y - pad_y, plot_max_y + pad_y)

    ax.set_title("Generated EEF Pose Grid")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    plt.show()

def main():
    if GRID_TYPE not in (1, 2):
        raise ValueError("GRID_TYPE must be 1 or 2")

    base_pose = read_pose_by_name(INPUT_CSV, INPUT_POSE_NAME)

    rows = create_grid_poses(
        base_pose=base_pose,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        offset_x=OFFSET_X,
        offset_y=OFFSET_Y,
        storage_id=STORAGE_ID,
        grid_type=GRID_TYPE,
    )

    write_poses(OUTPUT_CSV, rows)

    print(f"Created {len(rows)} grid poses from '{INPUT_POSE_NAME}'")
    print(f"Storage ID: {STORAGE_ID}")
    print(f"Output CSV: {OUTPUT_CSV}")

    plot_grid(
        rows=rows,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        grid_type=GRID_TYPE,
        offset_x=OFFSET_X,
        offset_y=OFFSET_Y,
    )


if __name__ == "__main__":
    main()