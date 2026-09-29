#!/usr/bin/env python3
"""Open-loop heave baseline: command a bladder volume, measure terminal vz.

Phase-0 measurement tool for the lake-data calibration. Drives the Gazebo
BuoyancyEngine directly through the model's parameter bridge (no HAL, no
controllers) and fits dz/dt from ground-truth odometry *position* (the
odometry twist is child-frame — with the NED spawn roll=pi its sign is
ambiguous, so we regress world-frame pose.position.z instead).

Usage (sim already running, workspace sourced):

    /usr/bin/python3 heave_baseline.py --volume 0.0012 \
        --out sim_data/heave_baseline.csv

Appends one CSV row per invocation:
    volume_m3, vz_mps, r2, n, z_start, z_end, settle_s, window_s
"""

import argparse
import csv
import os
import statistics
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64

BUOYANCY_CMD = "/model/glider_nautilus/buoyancy_engine"
BUOYANCY_VOL = "/model/glider_nautilus/buoyancy_engine/current_volume"
ODOMETRY = "/model/glider_nautilus/odometry"

VOLUME_TOL_M3 = 2e-5  # BuoyancyEngine ramp considered complete inside this


class HeaveProbe(Node):
    def __init__(self, target_volume: float):
        super().__init__("heave_baseline_probe")
        self.target_volume = target_volume
        self.current_volume = None
        self.samples = []  # (sim_time_s, z)
        self.recording = False
        self.cmd_pub = self.create_publisher(Float64, BUOYANCY_CMD, 10)
        self.create_subscription(Float64, BUOYANCY_VOL, self._on_volume, 10)
        self.create_subscription(Odometry, ODOMETRY, self._on_odom, 10)
        self.create_timer(0.2, self._publish_cmd)

    def _publish_cmd(self):
        self.cmd_pub.publish(Float64(data=float(self.target_volume)))

    def _on_volume(self, msg: Float64):
        self.current_volume = msg.data

    def _on_odom(self, msg: Odometry):
        if not self.recording:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.samples.append((t, msg.pose.pose.position.z))

    def ramp_complete(self) -> bool:
        return (
            self.current_volume is not None
            and abs(self.current_volume - self.target_volume) < VOLUME_TOL_M3
        )


def spin_for(node: Node, duration_s: float):
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)


def spin_until(node: Node, predicate, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return True
    return False


def linear_fit(samples):
    xs = [t for t, _ in samples]
    ys = [z for _, z in samples]
    try:
        slope = statistics.linear_regression(xs, ys).slope
        r2 = statistics.correlation(xs, ys) ** 2
    except statistics.StatisticsError:  # constant xs or ys
        return float("nan"), float("nan")
    return slope, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=float, required=True, help="bladder volume m^3")
    ap.add_argument("--settle", type=float, default=15.0)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--ramp-timeout", type=float, default=60.0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rclpy.init()
    node = HeaveProbe(args.volume)

    if not spin_until(node, lambda: node.current_volume is not None, 30.0):
        print("ERROR: no BuoyancyEngine volume feedback (sim up? bridge up?)")
        rclpy.shutdown()
        return 1
    if not spin_until(node, node.ramp_complete, args.ramp_timeout):
        print(
            f"ERROR: ramp to {args.volume} incomplete (current={node.current_volume})"
        )
        rclpy.shutdown()
        return 1

    spin_for(node, args.settle)
    node.samples.clear()
    node.recording = True
    spin_for(node, args.window)
    node.recording = False

    if len(node.samples) < 20:
        print(f"ERROR: only {len(node.samples)} odometry samples")
        rclpy.shutdown()
        return 1

    vz, r2 = linear_fit(node.samples)
    z_start, z_end = node.samples[0][1], node.samples[-1][1]

    new_file = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(
                [
                    "volume_m3",
                    "vz_mps",
                    "r2",
                    "n",
                    "z_start",
                    "z_end",
                    "settle_s",
                    "window_s",
                ]
            )
        w.writerow(
            [
                args.volume,
                f"{vz:.5f}",
                f"{r2:.4f}",
                len(node.samples),
                f"{z_start:.3f}",
                f"{z_end:.3f}",
                args.settle,
                args.window,
            ]
        )

    print(
        f"RESULT volume={args.volume:.5f} m^3  vz={vz:+.4f} m/s  "
        f"r2={r2:.3f}  z {z_start:.2f} -> {z_end:.2f}  n={len(node.samples)}"
    )
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
